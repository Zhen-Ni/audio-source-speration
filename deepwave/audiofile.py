#!/usr/bin/env python3

from __future__ import annotations
import os
import json
import subprocess as sp
from concurrent import futures
from abc import ABC, abstractmethod
import typing

import numpy as np
import numpy.typing as npt
import torch

from .utils import ensure_dir

__all__ = ('FFMPEG_BIN', 'FFPROBE_BIN', 'AudioFile',
           'CompressedAudio', 'RawAudio', 'convert_compressed_to_raw',
           'audiofile', 'build_raw')

FFMPEG_BIN = 'ffmpeg'
FFPROBE_BIN = 'ffprobe'


def _ffmpeg_read_stream(path_or_stream: str | typing.IO[bytes],
                        seek_time: float | None = None,
                        query_duration: float | None = None,
                        samplerate: int | None = None,
                        map_streams: list[int] | int | None = None
                        ) -> npt.NDArray:
    "Read stream and convert it to raw int16_t bytes."
    is_file = isinstance(path_or_stream, str)
    command = [FFMPEG_BIN]
    command += ['-loglevel', 'panic']
    if is_file:
        command += ['-i', path_or_stream]
    else:
        command += ['-i', '-']           # Input will arrive from pipe
    if samplerate is not None:
        command += '-ar', str(samplerate)
    if seek_time is not None:
        command += '-ss', str(seek_time)
    if query_duration is not None:
        command += ['-t', str(query_duration)]
    if map_streams is not None:
        if not np.iterable(map_streams):
            map_streams = [map_streams]
        for idx in map_streams:
            command += ['-map', str(f'0:{idx}')]

    # Ask for a raw 32-bit sound output, see ffmpeg -formats
    command += ['-f', 'f32le']
    command += ['-']            # Output to pipe.

    p = sp.Popen(command,
                 stdin=sp.PIPE, stdout=sp.PIPE)
    if is_file:
        out, err = p.communicate()
    else:
        out, err = p.communicate(path_or_stream.read())
    return np.frombuffer(out, dtype=np.float32)


def _ffprobe_read_info(path_or_stream: str | typing.IO[bytes]) -> list:
    "Get media info and returns it in json format."
    if isinstance(path_or_stream, str):
        stdout_data = sp.check_output([
            FFPROBE_BIN, "-loglevel", "panic",
            str(path_or_stream),
            '-print_format', 'json', '-show_format', '-show_streams'
        ])
    else:
        stdout_data = sp.check_output([
            'ffprobe', "-loglevel", "panic",
            '-',
            '-print_format', 'json', '-show_format', '-show_streams'
        ], input=path_or_stream.read())
    return json.loads(stdout_data.decode('utf-8'))


def convert_audio_channels(wav: torch.Tensor,
                           channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels."""

    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError(
            'The audio file has less channels than requested but is not mono.')
    return wav


class AudioFile(ABC):
    @abstractmethod
    def streams(self) -> int:
        ...

    @abstractmethod
    def channels(self) -> int:
        ...

    @abstractmethod
    def samplerate(self) -> int:
        ...
        
    @abstractmethod
    def frames(self) -> int:
        ...
        
    @abstractmethod
    def read(self,
             offset: int | None = None,
             samples: int | None = None,
             ) -> torch.Tensor:
        ...


class CompressedAudio(AudioFile):
    "Read audio from any format supported by ffmpeg."

    def __init__(self, path_or_buffer: str | typing.IO[bytes],
                 streams: int | list[int] | None = None,
                 channels: int | None = None,
                 samplerate: int | None = None):
        # Cached value.
        self._info = None
        
        self._path_or_buffer = path_or_buffer
        self._streams: list[int]
        self.select_stremas(streams)
        self._channels = channels
        self._samplerate = samplerate

    def __repr__(self) -> str:
        features = [("path", self.path or '*buffer*')]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", self.streams()))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"{self.__class__.__name__}({features_str})"

    @property
    def is_file(self) -> bool:
        return isinstance(self._path_or_buffer, str)

    @property
    def is_buffer(self) -> bool:
        return not self.is_file

    @property
    def path(self) -> str | None:
        if self.is_file:
            return self._path_or_buffer
        else:
            return None

    @property
    def source_info(self):
        if self._info is None:
            self._info = _ffprobe_read_info(self._path_or_buffer)
        return self._info

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.source_info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def source_streams(self) -> int:
        return len(self._audio_streams)

    def source_channels(self, stream=0) -> int:
        return int(self.source_info['streams'][self._audio_streams[stream]]
                   ['channels'])

    def source_samplerate(self, stream=0) -> int:
        return int(self.source_info['streams'][self._audio_streams[stream]]
                   ['sample_rate'])

    def source_frames(self, stream=0) -> int:
        return int(self.source_info['streams'][self._audio_streams[stream]]
                   ['duration_ts'])

    def streams(self) -> int:
        return len(self._streams)

    def select_stremas(self, streams: int | list[int] | None):
        "Select output streams."
        if streams is None:
            streams = self.source_streams()
        if isinstance(streams, int):
            streams = list(range(streams))
        self._streams = streams

    def channels(self) -> int:
        if self._channels is None:
            return self.source_channels()
        return self._channels

    def set_channels(self, channels: int | None):
        "Set number of channels."
        self._channels = channels

    def samplerate(self) -> int:
        if self._samplerate is None:
            return self.source_samplerate()
        return self._samplerate

    def set_samplerate(self, samplerate: int):
        self._samplerate = samplerate

    def frames(self) -> int:
        return int(self.duration()) * self.samplerate()

    def duration(self, stream=None) -> float:
        # Duration given in header might be different from those
        # defined in each stream.
        if stream is None:
            return float(self.source_info['format']['duration'])
        return float(self.source_info['streams'][self._audio_streams[stream]]
                     ['duration'])

    def read(self,
             offset: int | None = None,
             samples: int | None = None,
             ) -> torch.Tensor:
        """Similar to facebookresearch's implementation, but uses pipe
        instead of temp files to do the conversion.

        Args:
            offset (int):  offset in frames or None if no offset is needed.
            samples (int): read samples in frames to extract or None to extract until the end.
            streams (slice, int or list): streams to extract, can be a single int, a list or
                a slice. If it is a slice or list, the output will be of size [S, C, T]
                with S the number of streams, C the number of channels and T the number of samples.
                If it is an int, the output will be [C, T].
            samplerate (int): if provided, will resample on the fly. If None, no resampling will
                be done. Original sampling rate can be obtained with :method:`samplerate`.
            channels (int): if 1, will convert to mono. We do not rely on ffmpeg for that
                as ffmpeg automatically scale by +3dB to conserve volume when playing on speakers.
                See https://sound.stackexchange.com/a/42710.
                Our definition of mono is simply the average of the two channels. Any other
                value will be ignored.

        """
        wavs = []
        samplerate = self.samplerate()
        channels = self.channels()
        for i, stream in enumerate(self._streams):
            seek_time = None if offset is None else offset / samplerate
            duration = None if samples is None else samples / samplerate
            wav_array = _ffmpeg_read_stream(self._path_or_buffer, seek_time,
                                            duration, samplerate,
                                            self._audio_streams[stream])
            wav = torch.tensor(wav_array)
            wav = wav.view(-1, self.source_channels(stream)).t()
            if channels is not None:
                wav = convert_audio_channels(wav, channels)
            wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        return wav


class RawAudio(AudioFile):
    def __init__(self,
                 path: str,
                 streams: int,
                 channels: int,
                 samplerate: int,
                 dtype: typing.Type[np.floating] = np.float32):
        self._path = path
        self._streams = streams
        self._channels = channels
        self._samplerate = samplerate
        self._dtype = dtype

    def __repr__(self) -> str:
        features = [("path", self.path)]
        features.append(("channels", self.channels()))
        features.append(("streams", self.streams()))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"RawAudio({features_str})"

    @property
    def path(self) -> str | None:
        return self._path

    def streams(self) -> int:
        return self._streams

    def set_stremas(self, streams: int):
        "Set number of streams."
        self._streams = streams

    def channels(self) -> int:
        return self._channels

    def set_channels(self, channels: int):
        "Set number of channels."
        self._channels = channels

    def frame_size(self) -> int:
        "Size of a frame in bytes."
        element_size = self._dtype().itemsize
        return element_size * self._streams * self._channels

    def frames(self) -> int:
        return os.stat(self._path).st_size // self.frame_size()

    def samplerate(self) -> int:
        return self._samplerate
    
    def read(self,
             offset: int | None = None,
             samples: int | None = None,
             ) -> torch.Tensor:
        frame_size = self.frame_size()
        offset = 0 if offset is None else offset * frame_size
        size = None if samples is None else samples * frame_size
        with open(self._path, 'rb') as f:
            f.seek(offset)
            data = f.read(size)
        wav = np.frombuffer(data, dtype=self._dtype)
        wav = wav.reshape(-1, self.channels(),
                          self.streams()).T
        return torch.tensor(wav)


def convert_compressed_to_raw(source: str,
                              destination: str,
                              streams: int | list[int] | None = None,
                              channels: int | None = None,
                              samplerate: int | None = None
                              ):
    audio = CompressedAudio(source, streams, channels, samplerate)
    data = audio.read()
    with open(destination, 'wb') as f:
        f.write(data.numpy().T.tobytes())
    return RawAudio(destination, audio.streams(), audio.channels(),
                    audio.samplerate())


def audiofile(path: str,
              streams: int | list[int] | None = None,
              channels: int | None = None,
              samplerate: int | None = None) -> AudioFile | None:
    """Return AudioFile instance if path is audio file else return None."""
    if not os.path.isfile(path):
        return None
    if path.endswith('.raw'):
        if not isinstance(streams, int):
            raise AttributeError('streams must be specified for raw audio')
        if not isinstance(channels, int):
            raise AttributeError('channels must be specified for raw audio')
        if not isinstance(samplerate, int):
            raise AttributeError('samplerate must be specified for raw audio')
        return RawAudio(path, streams, channels, samplerate)
    afile = CompressedAudio(path, streams, channels, samplerate)
    try:
        # Raises subprocess.CalledProcessError if ffmpeg
        # doesnot support the format.
        afile.source_info
    except sp.CalledProcessError:
        return None
    return afile


def build_raw(source: str, destination: str, workers: int = None):
    """"Convert audio files in directory `source` to `destination`."""
    if not os.path.isdir(source):
        raise NotADirectoryError(f'not a directory: {source}')
    ensure_dir(destination)
    if workers:
        p = futures.ProcessPoolExecutor(workers)
    for filename in os.listdir(source):
        inpath = os.path.join(source, filename)
        audio = audiofile(inpath)
        if isinstance(audio, CompressedAudio):
            if '.' in filename:
                filename = ''.join(filename.split('.')[:-1])
            filename += '.raw'
            outpath = os.path.join(destination, filename)
            if workers:
                p.submit(convert_compressed_to_raw, inpath, outpath)
            else:
                convert_compressed_to_raw(inpath, outpath)
    if workers:
        p.shutdown()
    return
    
