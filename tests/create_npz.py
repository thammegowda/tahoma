#!/usr/bin/env python
"""
This is a sample file to create an npz file for testing deserialization from cpp
"""
from pathlib import Path
import numpy as np
import yaml
import logging as log
import torch

MYDIR = Path(__file__).parent.resolve()
DATA = MYDIR / "data"
NPZ_FILE = DATA / "test.npz"
log.basicConfig(level=log.INFO)


def str_as_array(s) -> np.ndarray:
    """Some npz readers do not support strings e.g. cnpy
    So we save strings as byte arrays. Assume ut-8 encoding map code points to bytes.
    """
    return np.array(list(s.encode('utf-8')), dtype=np.uint8)


if __name__ == "__main__":
    state = {}
    state["int32_vec"] = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    state["float32_vec"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    state["int64_vec"] = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    state["float64_vec"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    state["int32_mat"] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    state["float32_mat"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    state["int64_mat"] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    state["float64_mat"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    state["int32_tensor"] = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    state["float32_tensor"] = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    meta = {}
    for key, value in state.items():
        meta[key] = value.sum().item()  # Check if the value is valid

    state ['meta.yml'] = str_as_array(yaml.dump(meta))
    log.info(f"Saving to {NPZ_FILE}")
    np.savez(NPZ_FILE, **state)
