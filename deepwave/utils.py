#!/usr/bin/env python3

from __future__ import annotations

import os
import gc
from io import BytesIO
import tempfile
from contextlib import contextmanager

import torch
from torchaudio.transforms import Fade


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def ceil(x: float) -> int:
    y = int(x)
    return y if x == y else y + 1


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
    if os.path.exists(path):
        if os.path.isdir(path):
            return True
        return False
    parent, current = os.path.split(path)
    if parent and not ensure_dir(parent):
        return False
    if not current:
        raise ValueError('empty path')
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


def apply_model(model: torch.nn.Module,
                mix: torch.Tensor,
                segment_frames: int,
                overlap_frames: int,
                nshifts: int,
                max_shift: int
                ) -> torch.Tensor:
    """Apply model to a given mixture. Use fade, and add segments
    together in order to add model segment by segment. The shift
    trick is also applied, which makes the model time equivariant
    and improves SDR by up to 0.2 points.

    Parameters
    ----------
    mix: torch.Tensor
        Tensor of the mixed music. Shape of mix is: [batch, channel, length].
    segment_frames: int
        Segment length.
    overlap_frames: int
        Overlap length.
    nshifts: int
        If > 0, will shift in time `mix` by a random amount between 0
        and `max_shift` samples and apply the oppositve shift to the
        output. This is repeated `shifts` time and all predictions
        are averaged. This effectively makes the model time
        equivariant and improves SDR by up to 0.2 points.
        (usually use 10 as indicated by the demucs paper)
    max_shift: int
        Maximum shifted samples, usually selected to let the max
        shift time to be equal to 0.5s.
    """
    device = mix.device
    batch, channels, length = mix.shape

    chunk_len = segment_frames + overlap_frames

    # Apply model to mix without chunking if no semgment_frames is
    # defined or mix is not long enough.
    if (segment_frames == 0) or (length <= chunk_len):
        with torch.no_grad():
            out = _apply_model_shifted(model, mix, nshifts, max_shift)
        return out

    start = 0
    end = chunk_len

    fade = Fade(fade_in_len=0, fade_out_len=overlap_frames,
                fade_shape='linear')

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = _apply_model_shifted(model, chunk, nshifts, max_shift)
        out = fade(out)
        if start == 0:
            sources = out.size(1)
            final = torch.zeros([batch, sources, channels, length],
                                device=device)
            final[:, :, :, start: end] += out
            fade.fade_in_len = overlap_frames
            start += chunk_len - overlap_frames
        else:
            final[:, :, :, start: end] += out
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def _apply_model_shifted(model: torch.nn.Module,
                         mix: torch.Tensor,
                         nshifts: int,
                         max_shift: int) -> torch.Tensor:
    """Apply `model` to `mix` by shifting it a random amount between 0
    and `max_shift` `shifts` times."""
    if nshifts == 0:
        return model(mix)
    length = mix.size(-1)
    mix = torch.nn.functional.pad(mix, (max_shift, max_shift))
    offsets = torch.randint(0, max_shift, [nshifts])
    for i, offset in enumerate(offsets):
        shifted = mix[..., offset: offset + length + max_shift]
        shifted_out = model(shifted)
        outi = shifted_out[...,
                           max_shift - offset: max_shift - offset + length]
        if i == 0:
            out = outi
        else:
            out += outi
    out /= nshifts
    return out
