#!/usr/bin/env python3

from __future__ import annotations
import os
import subprocess

import torch

from .audioset import AudioSet

__all__ = 'build_musdb', 'MusdbSet'


def download_dataset():
    if 'musdb18' not in os.listdir():
        subprocess.run(
            ['wget', 'https://zenodo.org/record/1117372/files/musdb18.zip'])


def unzip_dataset():
    subprocess.run(['unzip', 'musdb18.zip', '-d', 'musdb18'])


def split_valid():
    os.mkdir('musdb18/train/train')
    os.mkdir('musdb18/train/valid')
    for i, f in enumerate(sorted(sorted(os.listdir('musdb18/train')))):
        fullpath = os.path.join('musdb18/train', f)
        if i < 84:
            dest = 'musdb18/train/train/'
        else:
            dest = 'musdb18/train/valid'
        subprocess.run(['mv', fullpath, dest])


def build_musdb(path):
    original_path = os.getcwd
    try:
        os.chdir(path)
        download_dataset()
        unzip_dataset()
        split_valid()
    except Exception:
        os.chdir(original_path)
        raise


class MusdbSet(AudioSet):
    def __init__(self,
                 path: str,
                 samples: int | None = None,
                 stride: int | None = None):
        streams = 5
        channels = 2
        samplerate = 44100
        super().__init__(path, samples, stride,
                         streams, channels, samplerate)
