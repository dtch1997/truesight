"""Utilities for reading, writing files"""

# Taken from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/utils.py

import io
import json
import os

from typing import Any, Sequence


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def read_jsonl(f, mode="r") -> Any:
    """Load a .jsonl file into a list."""
    f = _make_r_io_base(f, mode)
    jlist = [json.loads(line) for line in f]
    f.close()
    return jlist


def write_jsonl(obj: Sequence[Any], f, mode="w"):
    """Write a list to a .jsonl file."""
    f = _make_w_io_base(f, mode)
    for line in obj:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()
