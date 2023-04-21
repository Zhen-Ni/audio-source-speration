#!/usr/bin/env python3

from __future__ import annotations
import time

import torch

from .audiofile import CompressedAudio, ffmpeg_write_stream
from .utils import apply_model

__all__ = 'Separator',

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLERATE = 44100
SEGMENT_FRAMES = SAMPLERATE * 10
OVERLAP_FRAMES = SAMPLERATE
NSHIFTS = 10
MAX_SHIFT = SAMPLERATE // 2


class Separator:
    def __init__(self, *,
                 device: torch.device | int | str | None = None,
                 segment_frames: int = SEGMENT_FRAMES,
                 overlap_frames: int = OVERLAP_FRAMES,
                 nshifts: int = NSHIFTS,
                 max_shift: int = MAX_SHIFT
                 ):
        # Use property setter to move model and loss function to target device
        self.device = DEVICE if device is None else device
        self.segment_frames = segment_frames
        self.overlap_frames = overlap_frames
        self.nshifts = nshifts
        self.max_shift = max_shift

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | int | str) -> Separator:
        self._device = torch.device(device)
        return self

    def separate(self,
                 model: torch.nn.Module,
                 wav: torch.Tensor,
                 ) -> torch.Tensor:
        t_start = time.time()
        model.to(self.device)
        mix = wav.to(self.device)
        sources = apply_model(model, mix,
                              self.segment_frames,
                              self.overlap_frames,
                              self.nshifts,
                              self.max_shift).squeeze(0)
        print(f'separate result: '
              f'wall time = {time.time()- t_start:.2f}s')
        return sources

    def separate_file(self,
                      model: torch.nn.Module,
                      filename: str,
                      overwrite: bool = False
                      ):
        audio = CompressedAudio(filename)
        filename = filename.rsplit('.', 1)[0]
        wav = audio.read()
        sources = self.separate(model, wav)
        drums, bass, other, vocals = sources.numpy()
        ffmpeg_write_stream(drums, f'{filename}-drums.mp3',
                            audio.samplerate(), overwrite)
        ffmpeg_write_stream(bass, f'{filename}-bass.mp3',
                            audio.samplerate(), overwrite)
        ffmpeg_write_stream(other, f'{filename}-other.mp3',
                            audio.samplerate(), overwrite)
        ffmpeg_write_stream(vocals, f'{filename}-vocals.mp3',
                            audio.samplerate(), overwrite)
        return
