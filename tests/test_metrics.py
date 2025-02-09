#!/usr/bin/env python

import pytest
from pathlib import Path
import subprocess as sp
import os
import logging as log
from itertools import zip_longest

log.basicConfig(level=log.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

CMAKE_BINARY_DIR = os.environ.get('CMAKE_BINARY_DIR', None)
if CMAKE_BINARY_DIR is None:
    raise ValueError('CMAKE_BINARY_DIR not set')
CMAKE_BINARY_DIR = Path(CMAKE_BINARY_DIR)
TAHOMA_BIN = CMAKE_BINARY_DIR / 'tahoma'
assert TAHOMA_BIN.exists(), f'{TAHOMA_BIN} does not exist'

MYDIR = Path(__file__).parent
TEST_DATA = MYDIR / 'data'
TEST_MODELS = MYDIR / 'models'

@pytest.mark.parametrize('langs,is_qe', [
    ('en-de', True),
    ('en-ru', False),
    #('en-de', False),
    #('en-ru', True),
])
def test_metricx_large(langs, is_qe, year=2024, size='large'):
    log.info(f"Testing metricx-{year} {size} {langs} qe={is_qe}")
    year = year % 2000
    if year == 23:
        variant = is_qe and "-qe" or ""
        version = 'v2p0'
    elif year == 24:
        variant = "-hybrid"
        version = 'v2p6'
    else:
        raise ValueError(f'Invalid year: {year}')

    model_id = f"metricx-{year}{variant}-{size}-{version}"
    model_dir = TEST_MODELS / model_id
    assert model_dir.exists(), f'{model_dir} does not exist'
    dataset = f'sample.wmt23.{langs}'

    model_file = model_dir / 'model.npz'
    vocab_file = model_dir / 'spiece.model'
    input_file = TEST_DATA / f'{dataset}.tsv'
    tag = f'{dataset}.score.{model_id}.{is_qe and "qe" or "ref"}'
    gold_file =  TEST_DATA / f'{tag}.expect.txt'  # scores from original implementation
    for f in [model_file, vocab_file, input_file, gold_file]:
        assert f.exists(), f'{f} does not exist'

    gold_lines = gold_file.read_text().strip().split('\n')
    assert len(gold_lines) > 0, 'No input lines'
    out_file = TEST_DATA / f'{tag}.out.txt'
    out_file.unlink(missing_ok=True)
    #FIXME: mini-batch > 1 produces 1-few line in output
    cmd = f'{TAHOMA_BIN} predict {is_qe and "-qe" or ""} -m {model_file} -v {vocab_file} -i {input_file} --maxi-batch 10 --mini-batch 1 > {out_file}'
    log.info(f"RUN:\n  {cmd}")

    sp_env = os.environ.copy()
    sp_env["CUDA_VISIBLE_DEVICES"] = "0"  # limit to single GPU;
    sp.check_call(cmd, shell=True, text=True, env=sp_env)
    out_lines = out_file.read_text().strip().split('\n')

    assert len(out_lines) == len(gold_lines), f'Output lines {len(out_lines)} != Gold lines {len(gold_lines)}'
    epsilon = 1e-3  # allow small difference
    errors = []
    for i, (out_line, gold_line) in enumerate(zip_longest(out_lines, gold_lines), start=1):
        got = float(out_line)
        expect = float(gold_line)
        if abs(got - expect) > epsilon:
            errors.append(f'Line {i}: {got} != {expect}')
    if errors:
        if len(errors) > 10:
            log.error(f'{len(errors)} errors found; showing first 10')
        for msg in errors[:10]:
            log.error(msg)
    assert not errors, f'Found {len(errors)} errors (epsilon={epsilon})'
