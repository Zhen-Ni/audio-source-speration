#!/usr/bin/env python3

import os
import tempfile
from contextlib import contextmanager


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



