import torch
import torchaudio.functional as F


def hop_length(sample_rate, hop_ms=10):
    return max(1, int(sample_rate * hop_ms / 1000))


def frame_rate(sample_rate, hop_ms=10):
    return sample_rate / hop_length(sample_rate, hop_ms)


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
    win = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
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
    fbanks = F.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="htk",
    )
    mel = torch.matmul(spec.squeeze(0).t().float(), fbanks.to(spec.device))
    logmel = torch.log(mel.clamp_min(eps))
    return logmel.contiguous()
