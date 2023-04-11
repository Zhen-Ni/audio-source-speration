#!/usr/bin/env python3

import typing
import json
import subprocess as sp

import numpy as np
import numpy.typing as npt
import torch


__all__ = 'FFMPEG_BIN', 'AudioFile', 'convert_audio_channels'

FFMPEG_BIN = 'ffmpeg'


def _read_stream(path_or_stream: str | typing.IO[bytes],
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


def _read_info(path_or_stream: str | typing.IO[bytes]) -> list:
    "Get media info and returns it in json format."
    if isinstance(path_or_stream, str):
        stdout_data = sp.check_output([
            'ffprobe', "-loglevel", "panic",
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


class AudioFile:
    "Read audio from any format supported by ffmpeg."

    def __init__(self, path_or_buffer: str | typing.IO[bytes]):
        self._path_or_buffer = path_or_buffer

        # Cached values.
        self._info = None

    def __repr__(self):
        features = [("path", self.path or '*buffer*')]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioFile({features_str})"

    @property
    def is_file(self) -> bool:
        return isinstance(self._path_or_buffer, str)

    @property
    def is_buffer(self) -> bool:
        return not self.is_file

    @property
    def path(self) -> str | None:
        return self._path_or_buffer if self.is_file else None

    @property
    def info(self):
        if self._info is None:
            self._info = _read_info(self._path_or_buffer)
        return self._info

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self):
        return len(self._audio_streams)

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]
                   ['channels'])

    def duration(self, stream=None) -> float:
        # Duration given in header might be different from those
        # defined in each stream.
        if stream is None:
            return float(self.info['format']['duration'])
        return float(self.info['streams'][self._audio_streams[stream]]
                     ['duration'])

    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]
                   ['sample_rate'])

    def duration_ts(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]
                   ['duration_ts'])

    def read(self,
             seek_time: float | None = None,
             duration: float | None = None,
             streams: int | list[int] | None = None,
             samplerate: int | None = None,
             channels=None,
             ) -> torch.Tensor:
        """Similar to facebookresearch's implementation, but uses pipe
        instead of temp files to do the conversion.

        Args:
            seek_time (float):  seek time in seconds or None if no seeking is needed.
            duration (float): duration in seconds to extract or None to extract until the end.
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
        if streams is None:
            streams = list(range(len(self)))
        single = not np.iterable(streams)
        if single:
            streams = [streams]
        wavs = []
        for stream in streams:
            wav_array = _read_stream(self._path_or_buffer, seek_time,
                                     duration, samplerate,
                                     self._audio_streams[stream])
            wav = torch.tensor(wav_array)
            wav = wav.view(-1, self.channels(stream)).t()
            if channels is not None:
                wav = convert_audio_channels(wav, channels)
            wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav


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
