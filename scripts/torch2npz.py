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
import hashlib

import torch
import numpy as np
import transformers
import huggingface_hub as hf_hub

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

def pth_to_npz(pth_file, config_file, npz_file, md5=False, model_id=None):
    log.info(f'Converting {pth_file} to {npz_file}')
    state = {}
    state = torch.load(pth_file, weights_only=True)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.numpy()
    model_type = None
    Path(npz_file).resolve().parent.mkdir(parents=True, exist_ok=True)
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
    meta['converted_from'] = str(pth_file).replace(str(Path.home()), '~')
    meta['conversion_runtime'] = runtime_versions()
    if md5:
        meta['original_md5'] = file_md5(pth_file)
    if model_id:
        meta['model_id'] = model_id
    state['meta.yml'] = str_as_array(yaml.dump(meta))
    log.info(f'Saving {npz_file}')
    np.savez(npz_file, **state)



def main():
    args = parse_args()
    assert args.output.name.endswith('.npz'), 'Output file must have .npz extension.'
    #hf_to_torchscript(args.model, args.output, args.type)
    model_path = hf_hub.hf_hub_download(repo_id=args.model, filename=args.file)
    config_path = hf_hub.hf_hub_download(repo_id=args.model, filename="config.json")
    model_id = "hf://" + args.model
    pth_to_npz(model_path, config_path, args.output, md5=False, model_id=model_id)

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
