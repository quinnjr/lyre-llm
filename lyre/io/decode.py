import torch
import torchaudio


class AudioDecodeError(RuntimeError):
    pass


def load_audio(path, sample_rate=44100, channels=2):
    waveform, sr = torchaudio.load(path)
    if waveform.ndim != 2 or waveform.shape[0] == 0:
        raise AudioDecodeError(f"unexpected waveform shape {tuple(waveform.shape)} for {path}")
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if channels == 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if channels == 2 and waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    return waveform, sample_rate


def load_mono(path, sample_rate=44100):
    waveform, sr = load_audio(path, sample_rate=sample_rate, channels=1)
    return waveform[0], sr


def save_wav(path, waveform, sample_rate=44100):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform, sample_rate)
