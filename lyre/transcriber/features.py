# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import functools

import torch
import torchaudio.functional as F


def hop_length(sample_rate, hop_ms=10):
    return max(1, int(sample_rate * hop_ms / 1000))


def frame_rate(sample_rate, hop_ms=10):
    return sample_rate / hop_length(sample_rate, hop_ms)


@functools.lru_cache(maxsize=32)
def _hann_window(n_fft, device, dtype):
    return torch.hann_window(n_fft, device=device, dtype=dtype)


@functools.lru_cache(maxsize=32)
def _mel_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, device):
    fbanks = F.melscale_fbanks(
        n_freqs=n_freqs,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="htk",
    )
    return fbanks.to(device)


def compute_features(
    waveform,
    sample_rate=44100,
    n_mels=229,
    n_fft=2048,
    f_min=30,
    f_max=16000,
    hop_ms=10,
    eps=1e-5,
):
    hop = hop_length(sample_rate, hop_ms)
    win = _hann_window(n_fft, waveform.device, waveform.dtype)
    spec = F.spectrogram(
        waveform.unsqueeze(0),
        pad=0,
        window=win,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        power=2.0,
        normalized=False,
    )
    fbanks = _mel_fbanks(
        n_fft // 2 + 1, float(f_min), float(f_max), n_mels, sample_rate, spec.device
    )
    mel = torch.matmul(spec.squeeze(0).t().float(), fbanks)
    logmel = torch.log(mel.clamp_min(eps))
    return logmel.contiguous()
