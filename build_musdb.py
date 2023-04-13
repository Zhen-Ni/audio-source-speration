#!/usr/bin/env python3

from deepwave import build_musdb, build_raw

if __name__ == '__main__':
    workers = 4
    # build_musdb('../')
    build_raw('../musdb18/train/train', '../musdb18raw/train/train', workers)
    build_raw('../musdb18/train/valid', '../musdb18raw/train/valid', workers)
    build_raw('../musdb18/test', '../musdb18raw/test', workers)
