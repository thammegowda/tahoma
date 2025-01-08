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
from dataclasses import dataclass
from typing import Dict

import torch
import numpy as np
import huggingface_hub as hf_hub


WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
CONFIG_NAME = "config.json"

log.basicConfig(level=log.INFO)


@dataclass
class Model:
    id: str
    weight: str = None
    shards: str = None
    vocab: str = None
    config: str = None
    parent: 'Entry' = None

KNWON_MODELS = [
    Model(id='Unbabel/XCOMET-XL', weight='checkpoints/model.ckpt', config='hparams.yaml',
        parent=Model(id='facebook/xlm-roberta-xl', vocab='sentencepiece.bpe.model', config=CONFIG_NAME)),
    Model(id='Unbabel/XCOMET-XXL', weight='checkpoints/model.ckpt', config='hparams.yaml',
        parent=Model(id='facebook/xlm-roberta-xxl', vocab='sentencepiece.bpe.model', config=CONFIG_NAME)),
    Model(id='google/metricx-24-hybrid-large-v2p6', weight=WEIGHTS_NAME, config=CONFIG_NAME,
        parent=Model(id='google/mt5-base', vocab='spiece.model')),
    Model(id='google/metricx-24-hybrid-xl-v2p6', shards=WEIGHTS_INDEX_NAME,  config=CONFIG_NAME,
        parent=Model(id='google/mt5-base', vocab='spiece.model')),
    Model(id='google/metricx-24-hybrid-xxl-v2p6', shards=WEIGHTS_INDEX_NAME,  config=CONFIG_NAME,
        parent=Model(id='google/mt5-base', vocab='spiece.model')),    
]

ENTRIES = {}
for e in KNWON_MODELS:
    assert e.id not in ENTRIES, f'Entry {e.id} already exists.'
    ENTRIES[e.id] = e


def id2entry(model_id: str) -> Model:
    if model_id not in ENTRIES:
        return ValueError(f'Model {model_id} not supported. If this is a new model, please edit me (__file__) and add the mappings to the KNOWN_MODELS.')
    return ENTRIES[model_id]

class TaskType:
    TEXT_GENERATION = "text-generation"

def json_to_yaml(json_str) -> str:
    return yaml.dump(json.loads(json_str))


def hf_get(*args, **kwargs) -> Path:
    file = hf_hub.hf_hub_download(*args, **kwargs)
    return Path(file).absolute()

def load_hf_config(*args, **kwargs) -> dict:
    config_file = hf_get(*args, **kwargs)
    config_fmt = config_file.suffix.lower()
    config_txt = Path(config_file).read_text()
    if config_fmt in ('.yml', '.yaml'):
        return yaml.safe_load(config_txt)
    elif config_fmt == '.json':
        return json.loads(config_txt)
    else:
        raise NotImplementedError(f'Unsupported config format {config_fmt}')

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
    import transformers
    assert not out_file.exists(), f'File {out_file} already exists. Not overwriting.'
    assert task_type == TaskType.TEXT_GENERATION, f'Only {TaskType.TEXT_GENERATION} is supported.'
    pipe = transformers.pipeline(model=model_id, framework="pt")
    examples = []
    if task_type == TaskType.TEXT_GENERATION:
        ex = pipe.tokenizer("Hello, world! How are", return_tensors="pt")
        ex['decoder_input_ids'] = torch.zeros_like(ex['input_ids'])[:, :1]
        ex['return_dict'] = True
        #examples.append((ex["input_ids"], ex["attention_mask"]))
        examples.append(ex)
    else:
        raise NotImplementedError(f'Unsupported task type: {task_type}')

    #log.info(f'torch.compile on model')
    #model = torch.compile(model)
    #print(model(**examples[0]))

    scripted_model = torch.jit.script(model, example_inputs=examples)
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

def load_torch_weights_sharded(model_id, index_filename=WEIGHTS_INDEX_NAME, map_location=None):
    """
    HF's impl directly loads weights to nn.Module subclass. But we need state without binding to nn.Module.
    So we are rewriting the function here. See the link for original implementation:
    https://github.com/huggingface/transformers/blob/ca03842cdcf2823301171ab27aec4b6b1cafdbc1/src/transformers/modeling_utils.py#L408
    """
    index_file = hf_hub.hf_hub_download(repo_id=model_id, filename=index_filename)
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

def load_torch_state(model: Model) -> dict:
    try:
        if model.weight:
            log.info(f"Loading {model.id} / {model.weight} ")
            weights_file = hf_hub.hf_hub_download(repo_id=model.id, filename=model.weight)
            return load_torch_weights(weights_file)
        elif model.shards:
            log.info(f"Loading {model.id} / {model.shards} and the shards")
            return load_torch_weights_sharded(model.id, index_filename=model.shards)
        else:
            raise Exception(f'No weights file found for {model.id}')
    except Exception as e:
        if "Unauthorized" in str(e):
            log.warning(f'Login maybe required for {model.id}. Try "huggingface-cli login"', e)
            raise e

def pth_to_npz(args):
    model = args.model
    log.info(f'Converting {model.id} to {args.output}')

    model_type = None
    npz_file: Path = args.output
    npz_file.resolve().parent.mkdir(parents=True, exist_ok=True)
    config_obj = {}
    state = {}
    state["model_id"] = "hf://" + model.id
    if model.config:
        try:
            config_obj = load_hf_config(repo_id=model.id, filename=model.config)
            config_obj['model_id'] = model.id
            if config_obj.get('architectures'):
                model_type = config_obj['architectures'][0]
            else:
                model_type = model.id 
            if model.parent and model.parent.config:
                config_obj['parent'] = load_hf_config(repo_id=model.parent.id, filename=model.parent.config) 

            config_obj = dict(
                model=dict(
                    name=model_type,
                    args=config_obj
                )
            )
            config_obj['original_toolkit'] = 'huggingface-transformers'
            config_yml = yaml.dump(config_obj)
            state['config.yml'] = str_as_array(config_yml)
            config_out = npz_file.with_name('config.yml')
            log.info(f"Config file saved at {config_out}")
            config_out.write_text(config_yml)
        except Exception as e:
            log.warning(f'Config file {model.config} could not be loaded or parsed from {model.id}', e)
    
    weights = load_torch_state(model)
    assert weights
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            weights[k] = v.numpy()
    
    state.update(weights)
    log.info(f'Saving {npz_file}')
    np.savez(npz_file, **state)


def main():
    args = parse_args()
    model = args.model

    
    if args.format == 'npz':
        if not args.output.name.endswith('.npz'):
            # assume it is a directory
            ext = args.output.name.split('.')[-1]
            assert ext.lower() not in ('.pth', '.bin'), \
                f'Output file should be .npz extension. Avoid {ext}. Or just use directory and let me create the file inside it.'
            args.output = args.output / "model.npz"
            log.info(f'Output file is {args.output}')
        pth_to_npz(args)
    elif args.format == 'torchscript':
        if not args.output.name.endswith('.pth'):
            # assume it is a directory
            ext = args.output.name.split('.')[-1]
            assert ext.lower() not in ('.npz', '.bin'), \
                f'Output file should be .pth extension. Avoid {ext}. Or just use directory and let me create the file inside it.'
            args.output = args.output / "model.pth"
            log.info(f'Output file is {args.output}')
        hf_to_torchscript(model.id, args.output, TaskType.TEXT_GENERATION)
    else:
        raise NotImplementedError(f'Unsupported format {args.format}')

    vocab = model
    if not vocab.vocab and model.parent and model.parent.vocab:
        vocab = model.parent  # get vocab from parent
    if vocab.vocab:
        try:
            src_vocab = hf_hub.hf_hub_download(repo_id=vocab.id, filename=vocab.vocab)
            dest_vocab = Path(args.output).with_name(vocab.vocab)
            log.info(f'Copying {src_vocab} to {dest_vocab}')
            dest_vocab.write_bytes(Path(src_vocab).resolve().read_bytes())
        except Exception as e:
            log.warning(f'File {vocab.id} / {vocab.vocab} not found in the model repository.\n{e}')


def parse_args():
    ids = "\n".join(ENTRIES.keys())
    epilog = f'Known models:\n{ids}'
    parser = argparse.ArgumentParser(
        description='Converts an huggingface model to .npz compatible format.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog)
    parser.add_argument('-m', '--model', metavar="HF_ID", type = id2entry,  required=True,
                        help='Hgginface model ID or local path to model Id. See known models below for the list of ids',
                       )
    parser.add_argument('-o', '--output', type=Path, help='Output file path', required=True)
    parser.add_argument('-f', '--format', choices=['npz', 'torchscript'], default='npz', help='Output format')
    return parser.parse_args()

if __name__ == '__main__':
    main()
