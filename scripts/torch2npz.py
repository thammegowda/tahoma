#!/usr/bin/env python

"""

Converts a PyTorch model to a numpy .npz file.  
Pytorch has its own serialization format, which is not compatible with other frameworks. 

After unsuccesfully trying to load a PyTorch model in C++ with torch::load() and torch::jit::load(),
I decided to convert the model to a npz format.

"""


from pathlib import Path
import argparse
import logging as log
import datetime
import sys
import yaml
import json
import functools
import hashlib
from collections import defaultdict

import torch
import numpy as np
import huggingface_hub as hf_hub
import transformers
from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, CONFIG_NAME

log.basicConfig(level=log.INFO)


class TaskType:
    TEXT_GENERATION = "text-generation"

def json_to_yaml(json_str) -> str:
    return yaml.dump(json.loads(json_str))

def runtime_versions() -> dict:
    tl_mods = {name.split('.')[0] for name, mod in sys.modules.items()}
    # TODO: exclude standard library modules
    stdlib_path = Path(sys.executable).parent.parent / 'lib'
    versions = {name: sys.modules[name].__version__ for name in tl_mods 
                if hasattr(sys.modules[name], '__version__')}
    versions['python'] = sys.version.split()[0]
    return versions

def file_md5(file_path) -> str:
    log.info(f'Calculating md5 for {file_path}')
    with open(file_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    log.info(f'MD5 {Path(file_path).name}: {md5}')
    return md5

#WARNING: This function is not working. It is just a draft.
def hf_to_torchscript(model_id, out_file, task_type):
    assert not out_file.exists(), f'File {out_file} already exists. Not overwriting.'
    assert task_type == TaskType.TEXT_GENERATION, f'Only {TaskType.TEXT_GENERATION} is supported.'
    pipe = transformers.pipeline(model=hf_model_id, framework="pt")
    examples = []
    if task_type == TaskType.TEXT_GENERATION:
        ex = pipe.tokenizer("Hello, world!", return_tensors="pt")
        examples.append((ex["input_ids"], ex["attention_mask"]))
    else:
        raise NotImplementedError(f'Unsupported task type: {task_type}')
    print(examples)
    scripted_model = torch.jit.script(pipe.model, example_inputs=examples)
    out_file.resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted_model, str(out_file))

def str_as_array(s) -> np.ndarray:
    """Some npz readers do not support strings e.g. cnpy
    So we save strings as byte arrays. Assume ut-8 encoding map code points to bytes.
    """
    return np.array(list(s.encode('utf-8')), dtype=np.uint8)



""" 
For some models, e.g. metricx and unbabel-comets, the model code is not part of transformers lib.
So, we cant directly load the model with transformers automodel APIs.
We use lower level APIs to load the model weights without instantiating the nn.Module object.
"""

load_torch_weights = functools.partial(torch.load, weights_only=True) 

def load_torch_weights_sharded(model_id, map_location=None):
    """
    HF's impl directly loads weights to nn.Module subclass. But we need state without binding to nn.Module.
    So we are rewriting the function here. See the link for original implementation:
    https://github.com/huggingface/transformers/blob/ca03842cdcf2823301171ab27aec4b6b1cafdbc1/src/transformers/modeling_utils.py#L408
    """
    index_file = hf_hub.hf_hub_download(repo_id=model_id, filename=WEIGHTS_INDEX_NAME)
    index = json.loads(Path(index_file).read_text())
    state = {}
    # index is key to shard
    shard2key = defaultdict(list)
    for k, v in index["weight_map"].items():
        shard2key[v].append(k)
    shard_names = list(sorted(shard2key.keys()))
    assert shard_names, f'No shards found in {index_file}'
    log.info(f'Loading {len(shard_names)} shards: {shard_names}')
    for shard_name in shard_names:
        log.info(f'Loading shard {shard_name} and expecting {len(shard2key[shard_name])} keys in it.')
        shard_file = hf_hub.hf_hub_download(repo_id=model_id, filename=shard_name)
        shard_state = load_torch_weights(shard_file, map_location=map_location)
        # only select designated keys in the shard
        for key in shard2key[shard_name]:
            state[key] = shard_state[key]
    return state

def load_torch_state(args) -> dict:
    try: 
        weights_file = hf_hub.hf_hub_download(repo_id=args.model, filename=WEIGHTS_NAME)
        return load_torch_weights(weights_file)
    except Exception as e:
        log.warning(f'Failed to load weights from {WEIGHTS_NAME}.', e)

    try:
        log.info(f'Attempting shard index {WEIGHTS_INDEX_NAME}')
        return load_torch_weights_sharded(args.model)
    except Exception as e:
        log.warning('Failed to load weights from sharded files.', e)

    raise Exception(f'Failed to load model weights from {args.model}. Tried {WEIGHTS_NAME} and {WEIGHTS_INDEX_NAME}.')


def pth_to_npz(args):
    log.info(f'Converting {args.model} to {args.output}')
    state = {}
    state = load_torch_state(args)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.numpy()
    model_type = None
    npz_file: Path = args.output
    npz_file.resolve().parent.mkdir(parents=True, exist_ok=True)
    config_file = None
    model_id = "hf://" + args.model
    try:
        # note: unbabel uses hparams name
        config_file = hf_hub.hf_hub_download(repo_id=args.model, filename=CONFIG_NAME)
    except:
        log.warning(f'Config file {CONFIG_NAME} not found in the model repository')
    if config_file:
        config_file = Path(config_file)
        assert config_file.exists(), f'Config file {config_file} does not exist.'
        assert 'config.yml' not in state, 'config.yml already exists in the state dict.'
        config_obj = json.loads(config_file.read_text())
        if model_id:
            assert 'model_id' not in config_obj, 'model_id already exists in the config file.'
            config_obj['model_id'] = model_id
        if config_obj.get('architectures'):
            model_type = config_obj['architectures'][0]

        config_obj = dict(
            model=dict(
                name=model_type,
                args=config_obj
            )
        )
        config_obj['original_toolkit'] = 'huggingface-transformers'
        config_yml = yaml.dump(config_obj)
        state['config.yml'] = str_as_array(config_yml)
        Path(npz_file.with_name('config.yml')).write_text(config_yml)

    assert 'meta.yml' not in state, 'meta.json already exists in the state dict.'
    meta = {}
    meta['converted_at'] = str(datetime.datetime.now())
    meta['conversion_runtime'] = runtime_versions()
    meta['model_id'] = model_id
    state['meta.yml'] = str_as_array(yaml.dump(meta))
    log.info(f'Saving {npz_file}')
    np.savez(npz_file, **state)


def main():
    args = parse_args()
    
    if not args.output.name.endswith('.npz'):
        # assume it is a directory
        ext = args.output.name.split('.')[-1]
        assert ext.lower() not in ('.pth', '.bin'), \
            f'Output file should be .npz extension. Avoid {ext}. Or just use directory and let me create the file inside it.'
        args.output = args.output / "model.npz"
        log.info(f'Output file is {args.output}')
    #hf_to_torchscript(args.model, args.output, args.type)
    pth_to_npz(args)

    vocab_file_names = ['spiece.model', 'sentencepiece.bpe.model']
    for name in vocab_file_names:
        try:
            src_vocab = hf_hub.hf_hub_download(repo_id=args.model, filename=name)
            dest_vocab = Path(args.output).with_name(name)
            log.info(f'Copying {src_vocab} to {dest_vocab}')
            dest_vocab.write_bytes(Path(src_vocab).resolve().read_bytes())
            break
        except:
            log.warning(f'File {name} not found in the model repository.')

def parse_args():
    parser = argparse.ArgumentParser(description='Converts an huggingface model to .npz compatible format.')
    parser.add_argument('-m', '--model', type=str, help='Hgginface model ID or local path to model Id', required=True)
    parser.add_argument('-f', '--file', type=str, help='File inside the HF model repository', default="pytorch_model.bin")
    parser.add_argument('-t', '--type', default="text-generation",
                        help='Pipeline type. Used to make example input for tracing torchscript from the pipeline.')
    parser.add_argument('-o', '--output', type=Path, help='Output file path', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main()
