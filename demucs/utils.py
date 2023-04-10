#!/usr/bin/env python3

from __future__ import annotations

import os
from io import BytesIO
import tempfile
from contextlib import contextmanager

import torch


class AverageMeter:
    """Computes and stores the current and weighted average value."""
    __slots__ = 'average', 'value', 'sum', 'count'

    def __init__(self):
        self.average = 0.
        self.value = 0.
        self.sum = 0.
        self.count = 0

    def update(self, value: float, n: int = 1) -> AverageMeter:
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
        return self


@contextmanager
def temp_filenames(count, delete=True, **kwargs):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def ensure_dir(path: str) -> bool:
    """Ensure path exists.

    If path does not exist, try creating it. Return True if path
    exists, return False if path can not be created.
    """
    parent, current = os.path.split(path)
    if parent and not ensure_dir(parent):
        return False
    if not current:
        raise ValueError('empty path')
    if os.path.exists(path):
        if os.path.isdir(path):
            return True
        return False
    os.mkdir(path)
    return True


def copy_to(data, device: torch.device | str | None = None):
    # Avoid useless copy in gpu.
    # See https://discuss.pytorch.org/t/how-to-make-a-copy-of-a-gpu-model-on-the-cpu/90955/4
    if device is None:
        return data
    memory = BytesIO()
    torch.save(data, memory, pickle_protocol=-1)
    memory.seek(0)
    data = torch.load(memory, map_location=device)
    memory.close()
    return data
