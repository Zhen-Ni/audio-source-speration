#!/usr/bin/env python3

import os
import subprocess as sp

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import AudioFile


__all__ = 'Audioset',


def is_audiofile(path: str) -> AudioFile | None:
    """Return AudioFile instance if path is audio file else return None."""
    if not os.path.isfile(path):
        return None
    afile = AudioFile(path)
    try:
        # Raises subprocess.CalledProcessError if ffmpeg
        # doesnot support the format.
        afile.info
    except sp.CalledProcessError:
        return None
    return afile


class Audioset(torch.utils.data.Dataset):
    def __init__(self, path: str,
                 samples: int | None = None,
                 stride: int | None = None):
        self._path = path
        self._samples = samples
        self._stride = stride if stride is not None else samples
        self._filelist: list[AudioFile] = []
        for f in sorted(os.listdir(self._path)):
            if afile := is_audiofile(os.path.join(self._path, f)):
                self._filelist.append(afile)

        if self._samples is None:
            self._cumulative_sizes = np.arange(len(self.filelist)) + 1
        else:
            sizes = []
            for audio in self._filelist:
                size = (audio.duration_ts() -
                        self._samples) // self._stride + 1
                sizes.append(size)
            self._cumulative_sizes = np.cumsum(sizes)

    def __len__(self):
        return self._cumulative_sizes[-1]

    def __getitem__(self, index) -> torch.Tensor:
        file_index = np.searchsorted(self._cumulative_sizes, index,
                                     side='right')
        local_index = (index - self._cumulative_sizes[file_index - 1] if
                       file_index else index)
        audio = self.filelist[file_index]
        sample_rate = audio.samplerate()
        seek_time = (local_index * self._stride / sample_rate if
                     self._stride else None)
        duration = self._samples and self._samples / sample_rate
        wav = audio.read(seek_time, duration)
        return wav

    @property
    def filelist(self) -> list[AudioFile]:
        return self._filelist
