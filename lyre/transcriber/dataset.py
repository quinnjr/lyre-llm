# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import Dataset

from lyre.errors import AudioDecodeError, LyreError
from lyre.io.decode import load_mono
from lyre.reporting import warn
from lyre.transcriber.features import compute_features, frame_rate, hop_length
from lyre.transcriber.targets import midi_to_frames

# Waveform/target cache budget per worker process. A 4-minute mono stem costs
# roughly 42 MB of waveform plus ~24 MB of targets, so the default holds a
# handful of tracks and is hard-bounded regardless of dataset size.
DEFAULT_CACHE_BYTES = 256 * 1024 * 1024

DEFAULT_AUGMENT = {
    "enabled": True,
    "gain_prob": 0.5,
    "gain_db": 6.0,
    "pitch_shift_prob": 0.3,
    "pitch_shift_steps": 2,
    "time_stretch_prob": 0.3,
    "time_stretch": 0.1,
    "spec_augment_prob": 0.5,
    "freq_mask_width": 16,
    "time_mask_width": 24,
}


def _augment_config(augment):
    merged = dict(DEFAULT_AUGMENT)
    if augment:
        unknown = set(augment) - set(DEFAULT_AUGMENT)
        if unknown:
            raise ValueError(f"unknown augment keys: {sorted(unknown)}")
        merged.update(augment)
    return merged


class StemDataset(Dataset):
    """Windowed (log-mel, targets) pairs for one stem index.

    Memory: only the decoded waveform and the frame targets of the most
    recently used tracks are cached, under an LRU byte budget
    (``cache_bytes``). The log-mel is computed per window, so no full-track
    spectrogram is ever retained.
    """

    def __init__(
        self,
        entries,
        window_frames=128,
        n_mels=229,
        n_notes=128,
        sample_rate=44100,
        hop_ms=10,
        n_fft=2048,
        f_min=30,
        f_max=16000,
        min_note=24,
        max_note=95,
        train=True,
        augment=None,
        cache_bytes=DEFAULT_CACHE_BYTES,
    ):
        self.entries = list(entries)
        self.window_frames = window_frames
        self.n_mels = n_mels
        self.n_notes = n_notes
        self.sample_rate = sample_rate
        self.hop_ms = hop_ms
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.min_note = min_note
        self.max_note = max_note
        self.train = train
        self.augment = _augment_config(augment)
        self.cache_bytes = int(cache_bytes)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        # One unreadable file in a six-figure index must not abort a training
        # run hours in with a DataLoader worker traceback. Bad entries are
        # skipped, warned about once each, and counted so the caller can
        # report how much of the corpus never made it into training.
        self.skipped = 0
        self._skipped_paths = set()

    def __len__(self):
        return len(self.entries)

    # ---- caching -----------------------------------------------------------

    def _cache_put(self, key, data):
        nbytes = int(
            data[0].numel() * data[0].element_size()
            + data[1].nbytes
            + data[2].nbytes
            + data[3].nbytes
        )
        if nbytes > self.cache_bytes:
            return  # single item larger than the budget: never retain it
        # Drop any existing entry for this key first; incrementing without
        # subtracting the old size would make _cache_bytes drift upward and
        # eventually evict the cache down to a single entry.
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_bytes -= old[1]
        self._cache[key] = (data, nbytes)
        self._cache_bytes += nbytes
        while self._cache_bytes > self.cache_bytes and len(self._cache) > 1:
            _, (_, evicted) = self._cache.popitem(last=False)
            self._cache_bytes -= evicted

    def _load(self, idx):
        entry = self.entries[idx]
        key = (idx, entry["audio"], entry["midi"])
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit[0]
        waveform, sr = load_mono(entry["audio"], sample_rate=self.sample_rate)
        fps = frame_rate(sr, self.hop_ms)
        frames = midi_to_frames(
            entry["midi"],
            n_notes=self.n_notes,
            min_note=self.min_note,
            max_note=self.max_note,
            fps=fps,
        )
        data = (waveform, frames["pitch"], frames["onset"], frames["velocity"])
        self._cache_put(key, data)
        return data

    # ---- windowing ---------------------------------------------------------

    def _audio_frames(self, waveform, hop):
        # compute_features uses a centred STFT: floor(n / hop) + 1 frames.
        return waveform.numel() // hop + 1

    def _target_segments(self, start, src_frames, n_source):
        """Map each output frame to the half-open source span that feeds it.

        The mel is time-stretched by resampling (``F.interpolate``), so every
        source frame contributes to the output. Point-sampling the targets
        instead would skip source frames whenever src_frames > window_frames
        and silently delete the labels on them -- about 9% of onsets at rate
        1.1, while the mel still shows those onsets.

        Returns ``(lo, end, valid)``: per-output-frame start offsets into the
        padded source array, and a mask of frames whose source span exists at
        all. The spans tile contiguously, so ``np.maximum.reduceat`` over
        ``lo`` pools each of them. When src_frames <= window_frames every span
        is one frame wide and this reduces to the plain point sample.

        ``lo`` is clipped to ``n_source``, one past the last real frame, and
        ``_gather_targets`` appends a zero sentinel row so that index is legal
        for ``reduceat``. Clipping to ``n_source - 1`` instead would make the
        overrunning frame pool ``arr[lo : n_source - 1]``, silently dropping
        the label -- and any onset -- on the final labelled frame, which audio
        that outlasts the MIDI grid hits on every track.
        """
        rate = src_frames / self.window_frames
        j = np.arange(self.window_frames + 1, dtype=np.float64)
        bounds = start + np.floor(j * rate).astype(np.int64)
        lo = bounds[:-1]
        hi = np.maximum(bounds[1:], lo + 1)
        valid = lo < n_source
        lo = np.clip(lo, 0, n_source)
        end = max(int(lo[-1]) + 1, min(int(hi[-1]), n_source))
        return lo, end, valid

    def _gather_targets(self, arr, lo, end, valid):
        # One zero sentinel row so index n_source is a legal reduceat start.
        # Frames that reach it are masked out by `valid` regardless.
        padded = np.concatenate([arr[:end], np.zeros((1,) + arr.shape[1:], dtype=arr.dtype)])
        out = np.maximum.reduceat(padded, lo, axis=0)
        out = np.where(valid.reshape((-1,) + (1,) * (out.ndim - 1)), out, 0.0)
        return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))

    def __getitem__(self, idx):
        """Return the window for ``idx``, or the next readable one after it.

        Decoding happens inside a DataLoader worker, so an unreadable file
        would otherwise surface as a worker crash mid-epoch. Each bad entry is
        reported once and stepped over; only a wholly unreadable index is
        fatal, because that is a broken corpus rather than a broken file.
        """
        total = len(self.entries)
        if total == 0:
            raise IndexError("StemDataset has no entries")
        for offset in range(total):
            cur = (int(idx) + offset) % total
            try:
                return self._item(cur)
            except (AudioDecodeError, OSError, ValueError) as exc:
                self._record_skip(cur, exc)
        raise LyreError(
            "StemDataset: every one of the %d entries failed to load" % total
        )

    def _record_skip(self, idx, exc):
        entry = self.entries[idx]
        path = entry.get("audio") or entry.get("midi")
        if path in self._skipped_paths:
            return
        self._skipped_paths.add(path)
        self.skipped = len(self._skipped_paths)
        warn("skipping unreadable dataset entry %s: %s" % (path, exc))

    def _item(self, idx):
        waveform, pitch_np, onset_np, velocity_np = self._load(idx)
        hop = hop_length(self.sample_rate, self.hop_ms)
        n_frames = self._audio_frames(waveform, hop)
        augment_cfg = self.augment
        augmenting = self.train and augment_cfg["enabled"]

        rng = torch.Generator()
        if self.train:
            seed = hash((idx, int(torch.randint(1 << 30, (1,)).item()))) & (2**32 - 1)
        else:
            seed = hash(("val", idx)) & (2**32 - 1)
        rng.manual_seed(seed)

        # --- time stretch: choose how many source frames feed one output window
        rate = 1.0
        if augmenting and _bernoulli(augment_cfg["time_stretch_prob"], rng):
            span = float(augment_cfg["time_stretch"])
            rate = float(1.0 + (torch.rand(1, generator=rng).item() * 2 - 1) * span)
            rate = max(0.5, min(2.0, rate))
        src_frames = max(1, int(round(self.window_frames * rate)))

        max_start = max(0, n_frames - src_frames)
        if self.train:
            start = int(torch.randint(max_start + 1, (1,), generator=rng).item())
        else:
            # Deterministic centre window: the old code always scored the final
            # 1.28 s of every track (usually the decay tail).
            start = max_start // 2

        # --- waveform slice for this window
        s0 = start * hop
        n_needed = (src_frames - 1) * hop + self.n_fft
        seg = waveform[s0 : s0 + n_needed]
        if seg.numel() < self.n_fft:
            seg = F.pad(seg, (0, self.n_fft - seg.numel()))

        # --- waveform-domain augmentation
        n_steps = 0
        if augmenting and _bernoulli(augment_cfg["gain_prob"], rng):
            db = (torch.rand(1, generator=rng).item() * 2 - 1) * float(augment_cfg["gain_db"])
            seg = seg * float(10.0 ** (db / 20.0))
        if augmenting and augment_cfg["pitch_shift_steps"] and _bernoulli(augment_cfg["pitch_shift_prob"], rng):
            span = int(augment_cfg["pitch_shift_steps"])
            n_steps = int(torch.randint(-span, span + 1, (1,), generator=rng).item())
            if n_steps:
                seg = AF.pitch_shift(seg, self.sample_rate, n_steps)

        mel = compute_features(
            seg,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            f_min=self.f_min,
            f_max=self.f_max,
            hop_ms=self.hop_ms,
        )
        floor = float(mel.min())
        if mel.shape[0] < src_frames:
            mel = F.pad(mel, (0, 0, 0, src_frames - mel.shape[0]), value=floor)
        mel = mel[:src_frames]

        # --- targets on the same (possibly stretched) time map
        lo, end, valid = self._target_segments(start, src_frames, pitch_np.shape[0])
        pitch = self._gather_targets(pitch_np, lo, end, valid)
        onset = self._gather_targets(onset_np, lo, end, valid)
        velocity = self._gather_targets(velocity_np, lo, end, valid)

        # --- time stretch on the mel time axis (pitch-preserving, unlike a
        # resample of the waveform, so the pitch targets stay valid)
        if src_frames != self.window_frames:
            mel = (
                F.interpolate(
                    mel.t().unsqueeze(0), size=self.window_frames, mode="linear", align_corners=False
                )
                .squeeze(0)
                .t()
                .contiguous()
            )

        # --- keep the pitch targets consistent with a pitch-shifted waveform
        if n_steps:
            pitch = _shift_notes(pitch, n_steps)
            velocity = _shift_notes(velocity, n_steps)

        if mel.shape[0] < self.window_frames:
            mel = F.pad(mel, (0, 0, 0, self.window_frames - mel.shape[0]), value=floor)
        mel = mel[: self.window_frames]

        if augmenting and _bernoulli(augment_cfg["spec_augment_prob"], rng):
            mel = _spec_augment(mel, rng, augment_cfg["freq_mask_width"], augment_cfg["time_mask_width"])
        return mel, pitch, onset, velocity


def _bernoulli(prob, rng):
    return torch.rand(1, generator=rng).item() < float(prob)


def _shift_notes(target, n_steps):
    """Transpose a (frames, n_notes) target by n_steps semitones, no wrap."""
    out = torch.zeros_like(target)
    n_notes = target.shape[1]
    if abs(n_steps) >= n_notes:
        return out
    if n_steps > 0:
        out[:, n_steps:] = target[:, : n_notes - n_steps]
    else:
        out[:, : n_notes + n_steps] = target[:, -n_steps:]
    return out


def _spec_augment(mel, rng, freq_mask_width=16, time_mask_width=24):
    mel = mel.clone()
    n_mels = mel.shape[1]
    time_frames = mel.shape[0]
    floor = mel.min()
    if freq_mask_width > 0 and torch.randint(2, (1,), generator=rng).item():
        f0 = int(torch.randint(0, n_mels, (1,), generator=rng).item())
        width = int(torch.randint(1, int(freq_mask_width) + 1, (1,), generator=rng).item())
        mel[:, f0 : f0 + width] = floor
    if time_mask_width > 0 and torch.randint(2, (1,), generator=rng).item():
        t0 = int(torch.randint(0, time_frames, (1,), generator=rng).item())
        width = int(torch.randint(1, int(time_mask_width) + 1, (1,), generator=rng).item())
        mel[t0 : t0 + width] = floor
    return mel
