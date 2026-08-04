import torch
from torch.utils.data import Dataset

from lyre.io.decode import load_mono
from lyre.transcriber.features import compute_features
from lyre.transcriber.targets import midi_to_frames


class StemDataset(Dataset):
    def __init__(
        self,
        entries,
        window_frames=128,
        n_mels=229,
        n_notes=128,
        sample_rate=44100,
        hop_ms=10,
        train=True,
    ):
        self.entries = list(entries)
        self.window_frames = window_frames
        self.n_mels = n_mels
        self.n_notes = n_notes
        self.sample_rate = sample_rate
        self.hop_ms = hop_ms
        self.train = train
        self._cache = {}

    def __len__(self):
        return len(self.entries)

    def _load(self, idx):
        entry = self.entries[idx]
        key = (idx, entry["audio"], entry["midi"])
        if key in self._cache:
            return self._cache[key]
        waveform, sr = load_mono(entry["audio"], sample_rate=self.sample_rate)
        frames = midi_to_frames(
            entry["midi"], n_notes=self.n_notes, fps=100
        )
        mel = compute_features(
            waveform, sample_rate=sr, n_mels=self.n_mels, hop_ms=self.hop_ms
        )
        data = (mel, frames["pitch"], frames["onset"], frames["velocity"], sr)
        self._cache[key] = data
        return data

    def __getitem__(self, idx):
        mel, pitch, onset, velocity, sr = self._load(idx)
        rng = torch.Generator()
        rng.manual_seed(hash((idx, torch.randint(1 << 30, (1,)).item())) & (2**32 - 1))
        if self.train and mel.shape[0] > self.window_frames:
            start = int(torch.randint(mel.shape[0] - self.window_frames, (1,), generator=rng).item())
        else:
            start = max(0, mel.shape[0] - self.window_frames)
        end = min(start + self.window_frames, mel.shape[0])
        mel = mel[start:end]
        pitch = torch.from_numpy(pitch[start:end])
        onset = torch.from_numpy(onset[start:end])
        velocity = torch.from_numpy(velocity[start:end])
        if mel.shape[0] < self.window_frames:
            pad = self.window_frames - mel.shape[0]
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))
            pitch = torch.nn.functional.pad(pitch, (0, 0, 0, pad))
            onset = torch.nn.functional.pad(onset, (0, 0, 0, pad))
            velocity = torch.nn.functional.pad(velocity, (0, 0, 0, pad))
        if self.train:
            mel = _spec_augment(mel, rng)
        return mel, pitch, onset, velocity


def _spec_augment(mel, rng):
    mel = mel.clone()
    n_mels = mel.shape[1]
    time_frames = mel.shape[0]
    if torch.randint(2, (1,), generator=rng).item():
        f0 = int(torch.randint(0, n_mels, (1,), generator=rng).item())
        width = int(torch.randint(1, 16, (1,), generator=rng).item())
        mel[:, f0 : f0 + width] = mel.min()
    if torch.randint(2, (1,), generator=rng).item():
        t0 = int(torch.randint(0, time_frames, (1,), generator=rng).item())
        width = int(torch.randint(1, 24, (1,), generator=rng).item())
        mel[t0 : t0 + width] = mel.min()
    return mel
