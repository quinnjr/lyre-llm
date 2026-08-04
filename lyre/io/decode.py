# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import torch
import torchaudio

from lyre.errors import AudioDecodeError

__all__ = ["AudioDecodeError", "load_audio", "load_mono", "save_wav"]


def _fit_channels(waveform, channels):
    if channels is None:
        return waveform
    have = waveform.shape[0]
    if have == channels:
        return waveform
    if have > channels:
        if channels == 1:
            return waveform.mean(dim=0, keepdim=True)
        # Down-mix the surplus channels evenly into the ones we keep so a
        # 6-channel file does not silently pass through un-narrowed. Each
        # output channel is a convex combination of all `have` inputs -- its
        # own with weight channels/have, every surplus channel with weight
        # 1/have -- so the weights sum to 1. That means the result can never
        # exceed the input peak (no clipping on re-encode), and identical
        # content on every channel maps to itself, so the same material
        # delivered as 5.1 and as stereo produces the same loudness and the
        # same log-mel. Simply adding a fraction of the surplus on top of
        # full-amplitude kept channels does neither.
        head = waveform[:channels]
        tail = waveform[channels:]
        return (head * channels + tail.sum(dim=0, keepdim=True).expand_as(head)) / have
    # have < channels: replicate what we have to fill the requested layout.
    reps = [waveform[i % have] for i in range(channels)]
    return torch.stack(reps, dim=0)


def load_audio(path, sample_rate=44100, channels=2):
    """Decode ``path`` to a ``(channels, samples)`` tensor at ``sample_rate``.

    Any decoder failure is surfaced as :class:`~lyre.errors.AudioDecodeError`.
    """
    try:
        waveform, sr = torchaudio.load(path)
    except Exception as exc:
        raise AudioDecodeError(f"cannot decode audio from {path}: {exc}") from exc
    if not torch.is_tensor(waveform) or waveform.ndim != 2 or waveform.shape[0] == 0:
        raise AudioDecodeError(
            f"unexpected waveform shape {tuple(getattr(waveform, 'shape', ()))} for {path}"
        )
    if waveform.shape[1] == 0:
        raise AudioDecodeError(f"decoded zero samples from {path}")
    if sr <= 0:
        raise AudioDecodeError(f"invalid sample rate {sr} reported for {path}")
    if sr != sample_rate:
        try:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        except Exception as exc:
            raise AudioDecodeError(
                f"cannot resample {path} from {sr}Hz to {sample_rate}Hz: {exc}"
            ) from exc
    waveform = _fit_channels(waveform, channels)
    return waveform, sample_rate


def load_mono(path, sample_rate=44100):
    waveform, sr = load_audio(path, sample_rate=sample_rate, channels=1)
    return waveform[0], sr


def save_wav(path, waveform, sample_rate=44100):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform, sample_rate)
