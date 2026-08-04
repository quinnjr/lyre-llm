# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import numpy as np
import pretty_midi
import pytest
import torch
import torchaudio

from lyre.errors import LyreError
from lyre.transcriber.dataset import (
    DEFAULT_AUGMENT,
    StemDataset,
    _augment_config,
    _shift_notes,
)
from lyre.transcriber.metrics import frame_f1, onset_f1
from lyre.transcriber.model import MultiPitchNet, predict_track
from lyre.transcriber.targets import midi_to_frames

SR = 16000

NO_AUGMENT = dict(DEFAULT_AUGMENT, enabled=False)


def _write_midi(path):
    """Two notes: C4 at 0.00-0.50 (vel 100) and G4 at 1.00-1.20 (vel 64)."""
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=64, pitch=67, start=1.0, end=1.2))
    midi.instruments.append(inst)
    midi.write(str(path))
    return str(path)


def _write_sustained_midi(path, pitch=60, seconds=3.0, velocity=100):
    """One note covering the whole file, so any window sees exactly that pitch."""
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(
        pretty_midi.Note(velocity=velocity, pitch=pitch, start=0.0, end=seconds)
    )
    midi.instruments.append(inst)
    midi.write(str(path))
    return str(path)


def _tiny_dataset(entries, **kwargs):
    defaults = dict(
        window_frames=32,
        n_mels=32,
        n_fft=512,
        f_min=30,
        f_max=SR // 2,
        sample_rate=SR,
        train=False,
        augment={"enabled": False},
    )
    defaults.update(kwargs)
    return StemDataset(entries, **defaults)


def _write_wav(path, seconds):
    torch.manual_seed(0)
    samples = int(SR * seconds)
    wave = 0.1 * torch.sin(
        2 * torch.pi * 440.0 * torch.arange(samples, dtype=torch.float32) / SR
    )
    torchaudio.save(str(path), wave.unsqueeze(0), SR)
    return str(path)


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------


def test_midi_to_frames_known_two_note_file(tmp_path):
    frames = midi_to_frames(_write_midi(tmp_path / "a.mid"), fps=100)

    assert frames["duration"] == pytest.approx(1.2)
    assert frames["pitch"].shape == (121, 128)
    assert frames["onset"].shape == (121,)

    assert np.flatnonzero(frames["onset"]).tolist() == [0, 100]
    assert frames["onset"].sum() == 2.0

    # sustain maps: C4 for frames 0..49, G4 for frames 100..119
    assert frames["pitch"][:, 60].sum() == 50.0
    assert np.flatnonzero(frames["pitch"][:, 60]).tolist() == list(range(0, 50))
    assert frames["pitch"][:, 67].sum() == 20.0
    assert np.flatnonzero(frames["pitch"][:, 67]).tolist() == list(range(100, 120))
    assert frames["pitch"].sum() == 70.0

    # velocities are scaled 0..1 per note
    assert frames["velocity"][0, 60] == pytest.approx(100 / 127.0, abs=1e-6)
    assert frames["velocity"][49, 60] == pytest.approx(100 / 127.0, abs=1e-6)
    assert frames["velocity"][100, 67] == pytest.approx(64 / 127.0, abs=1e-6)
    assert frames["velocity"][50, 60] == 0.0


def test_midi_to_frames_short_duration_does_not_raise(tmp_path):
    # duration shorter than the MIDI used to walk off the end of the grid.
    path = _write_midi(tmp_path / "a.mid")
    frames = midi_to_frames(path, duration=0.5, fps=100)

    assert frames["pitch"].shape == (51, 128)
    assert frames["onset"].shape == (51,)
    # Only the first note survives; the 1.0 s note is past the grid.
    assert np.flatnonzero(frames["onset"]).tolist() == [0]
    assert frames["pitch"][:, 67].sum() == 0.0
    assert frames["pitch"][:, 60].sum() == 50.0


def test_midi_to_frames_respects_the_note_range(tmp_path):
    path = _write_midi(tmp_path / "a.mid")
    frames = midi_to_frames(path, min_note=61, max_note=95, fps=100)
    assert frames["pitch"][:, 60].sum() == 0.0
    assert frames["pitch"][:, 67].sum() == 20.0


@pytest.mark.parametrize(
    "min_note,max_note,n_notes",
    [
        (-1, 95, 128),      # negative floor
        (95, 24, 128),      # inverted range
        (24, 128, 128),     # max_note == n_notes indexes past the pitch axis
        (24, 200, 128),
        (0, 128, 128),
    ],
)
def test_midi_to_frames_rejects_an_out_of_range_note_window(tmp_path, min_note, max_note, n_notes):
    path = _write_midi(tmp_path / "a.mid")
    with pytest.raises(ValueError):
        midi_to_frames(path, min_note=min_note, max_note=max_note, n_notes=n_notes)


def test_midi_to_frames_accepts_the_boundary_window(tmp_path):
    path = _write_midi(tmp_path / "a.mid")
    frames = midi_to_frames(path, min_note=0, max_note=127, n_notes=128, fps=100)
    assert frames["pitch"][:, 60].sum() == 50.0


def test_midi_to_frames_fps_drives_the_grid(tmp_path):
    path = _write_midi(tmp_path / "a.mid")
    fast = midi_to_frames(path, fps=100)
    slow = midi_to_frames(path, fps=50)

    assert fast["pitch"].shape == (121, 128)
    assert slow["pitch"].shape == (61, 128)
    assert np.flatnonzero(fast["onset"]).tolist() == [0, 100]
    assert np.flatnonzero(slow["onset"]).tolist() == [0, 50]
    # 0.5 s of C4 is 50 frames at 100 fps and 25 at 50 fps.
    assert slow["pitch"][:, 60].sum() == 25.0


# --------------------------------------------------------------------------
# dataset: the time-stretch target map
#
# The mel is stretched by resampling, so every source frame contributes to the
# output. The targets used to be point-sampled with floor(), which skipped
# source frames whenever src_frames > window_frames and deleted the labels on
# them -- ~9% of onsets at rate 1.1. Shapes were unchanged and the loss still
# went down, so nothing showed.
# --------------------------------------------------------------------------


def _point_sample(arr, start, src_frames, window_frames):
    """The old (buggy) gather: one source frame per output frame, floor()."""
    rate = src_frames / window_frames
    idx = start + np.floor(np.arange(window_frames) * rate).astype(np.int64)
    return arr[idx]


def _gather(dataset, arr, start, src_frames, n_source=None):
    if n_source is None:
        n_source = arr.shape[0]
    lo, end, valid = dataset._target_segments(start, src_frames, n_source)
    return dataset._gather_targets(arr, lo, end, valid).numpy()


def _onset_curve(n_source, seed=0):
    """Irregularly spaced onsets, min gap 3, so no rate aliases with the grid."""
    rng = np.random.default_rng(seed)
    onset = np.zeros(n_source, dtype=np.float32)
    frame = 0
    while frame < n_source:
        onset[frame] = 1.0
        frame += int(rng.integers(3, 7))
    return onset


@pytest.mark.parametrize("src_frames", [129, 135, 141, 154, 192, 256])
@pytest.mark.parametrize("start", [0, 13, 77])
def test_time_stretch_preserves_every_onset_label(src_frames, start):
    window = 128
    dataset = _tiny_dataset([], window_frames=window)
    n_source = 600
    onset = _onset_curve(n_source)

    expected = int(onset[start : start + src_frames].sum())
    assert expected > 20  # the case is only meaningful if there is a lot to lose
    out = _gather(dataset, onset, start, src_frames)

    assert out.shape == (window,)
    assert int(out.sum()) == expected, (
        "the stretched window lost %d of %d onsets"
        % (expected - int(out.sum()), expected)
    )


def test_point_sampling_loses_the_onsets_the_max_pool_keeps():
    """The dropped-label bug, reproduced. Without it the fix is unfalsifiable."""
    window, src_frames = 128, 141  # rate 1.10, the measured case
    dataset = _tiny_dataset([], window_frames=window)

    total_expected = total_old = total_kept = 0
    for seed in range(60):
        onset = _onset_curve(600, seed=seed)
        total_expected += int(onset[:src_frames].sum())
        total_kept += int(_gather(dataset, onset, 0, src_frames).sum())
        total_old += int(_point_sample(onset, 0, src_frames, window).sum())

    assert total_kept == total_expected
    lost = (total_expected - total_old) / total_expected
    # rate 1.1 skips 13 of 141 source frames outright (9.2%); the onsets that
    # sat on them went with them.
    assert 0.04 <= lost <= 0.18, "point sampling lost %.1f%% of onsets" % (100 * lost)


def test_time_stretch_preserves_every_pitch_label():
    window = 128
    src_frames = 141  # rate ~1.10
    dataset = _tiny_dataset([], window_frames=window)
    n_source = 400
    pitch = np.zeros((n_source, 128), dtype=np.float32)
    # A different single pitch active on each source frame: any dropped frame
    # is a dropped label.
    for frame in range(n_source):
        pitch[frame, 40 + (frame % 30)] = 1.0

    out = _gather(dataset, pitch, 0, src_frames)
    assert out.shape == (window, 128)

    active_in = {40 + (f % 30) for f in range(src_frames)}
    active_out = set(np.flatnonzero(out.max(axis=0)).tolist())
    assert active_out == active_in
    # Every source frame's label survives into some output frame.
    assert int(out.sum()) >= src_frames - 0  # max-pooling never invents labels
    assert int(out.sum()) <= src_frames


@pytest.mark.parametrize("src_frames", [1, 17, 64, 100, 127, 128])
def test_rates_at_or_below_one_are_bit_identical_to_point_sampling(src_frames):
    """No degradation: compression and rate 1.0 must not change behaviour."""
    window = 128
    dataset = _tiny_dataset([], window_frames=window)
    rng = np.random.default_rng(0)
    n_source = 500
    onset = rng.random(n_source).astype(np.float32)
    pitch = rng.random((n_source, 128)).astype(np.float32)

    for start in (0, 7, 200):
        for arr in (onset, pitch):
            expected = _point_sample(arr, start, src_frames, window)
            got = _gather(dataset, arr, start, src_frames)
            assert np.array_equal(got, expected), (
                "src_frames=%d start=%d diverged from point sampling"
                % (src_frames, start)
            )


def _gather_oracle(arr, start, src_frames, window_frames, n_source):
    """Independent reference: max over each output frame's real source span."""
    rate = src_frames / window_frames
    out = np.zeros((window_frames,) + arr.shape[1:], dtype=np.float32)
    for j in range(window_frames):
        lo = start + int(np.floor(j * rate))
        hi = max(start + int(np.floor((j + 1) * rate)), lo + 1)
        lo = min(lo, n_source)
        hi = min(hi, n_source)
        if lo < hi:
            out[j] = arr[lo:hi].max(axis=0)
    return out


@pytest.mark.parametrize("src_frames", [1, 64, 128, 129, 141, 192, 256])
@pytest.mark.parametrize("overrun", [0, 1, 5, 40, 130])
def test_target_gather_matches_the_oracle_when_the_window_overruns_the_grid(
    src_frames, overrun
):
    """The window runs past the end of the MIDI grid on essentially every track.

    Audio outlasts the notated material, so the last window of a track always
    overruns. Clipping the segment starts to ``n_source - 1`` instead of
    ``n_source`` made the overrunning output frame pool ``arr[lo : n_source-1]``
    -- an empty span -- which silently deleted the label, and any onset, on the
    final labelled frame. Every shape stayed correct.
    """
    window = 128
    dataset = _tiny_dataset([], window_frames=window)
    n_source = 200
    rng = np.random.default_rng(11)
    onset = rng.random(n_source).astype(np.float32)
    pitch = rng.random((n_source, 128)).astype(np.float32)

    # Start so the window ends `overrun` frames past the last labelled frame.
    start = max(0, n_source + overrun - src_frames)
    for arr in (onset, pitch):
        got = _gather(dataset, arr, start, src_frames, n_source=n_source)
        expected = _gather_oracle(arr, start, src_frames, window, n_source)
        assert np.array_equal(got, expected), (
            "src_frames=%d overrun=%d start=%d" % (src_frames, overrun, start)
        )


def test_the_final_labelled_frame_survives_a_window_that_overruns_the_grid():
    # One label, on the very last frame of the MIDI grid, with the window
    # running past it. That frame is where a whole track's closing note lives.
    window = 128
    dataset = _tiny_dataset([], window_frames=window)
    n_source = 40
    onset = np.zeros(n_source, dtype=np.float32)
    onset[n_source - 1] = 1.0

    for src_frames in (129, 141, 192, 256):
        out = _gather(dataset, onset, 0, src_frames, n_source=n_source)
        assert out.shape == (window,)
        assert float(out.sum()) == 1.0, (
            "src_frames=%d lost the label on the final frame" % src_frames
        )


def test_target_gather_zeroes_frames_past_the_end_of_the_track():
    window = 128
    dataset = _tiny_dataset([], window_frames=window)
    n_source = 40
    onset = np.ones(n_source, dtype=np.float32)

    out = _gather(dataset, onset, 20, 141, n_source=n_source)
    # Source frames 20..39 exist; everything mapped past 39 is zero, not a
    # clamp onto the final frame.
    assert out.shape == (window,)
    assert float(out[-1]) == 0.0
    assert float(out[0]) == 1.0
    assert 0 < int(out.sum()) < window


# --------------------------------------------------------------------------
# dataset: augmentation
# --------------------------------------------------------------------------


def test_shift_notes_transposes_up_and_down():
    target = torch.zeros(4, 128)
    target[:, 60] = 1.0

    up = _shift_notes(target, 2)
    assert torch.equal(up[:, 62], torch.ones(4))
    assert float(up[:, 60].sum()) == 0.0
    assert float(up.sum()) == 4.0

    down = _shift_notes(target, -2)
    assert torch.equal(down[:, 58], torch.ones(4))
    assert float(down[:, 60].sum()) == 0.0
    assert float(down.sum()) == 4.0


def test_shift_notes_zero_steps_is_the_identity():
    torch.manual_seed(0)
    target = torch.rand(4, 128)
    assert torch.equal(_shift_notes(target, 0), target)


def test_shift_notes_preserves_values_not_just_positions():
    target = torch.zeros(3, 128)
    target[:, 70] = 0.375
    shifted = _shift_notes(target, 3)
    assert float(shifted[0, 73]) == 0.375


def test_shift_notes_does_not_wrap_at_the_edges():
    top = torch.zeros(2, 128)
    top[:, 127] = 1.0
    assert float(_shift_notes(top, 1).sum()) == 0.0
    assert float(_shift_notes(top, -1)[:, 126].sum()) == 2.0

    bottom = torch.zeros(2, 128)
    bottom[:, 0] = 1.0
    assert float(_shift_notes(bottom, -1).sum()) == 0.0
    assert float(_shift_notes(bottom, 1)[:, 1].sum()) == 2.0


@pytest.mark.parametrize("n_steps", [128, 129, -128, -400])
def test_shift_notes_beyond_the_pitch_axis_is_all_zeros(n_steps):
    torch.manual_seed(0)
    target = torch.rand(2, 128)
    out = _shift_notes(target, n_steps)
    assert out.shape == target.shape
    assert float(out.abs().sum()) == 0.0


def test_augment_config_rejects_unknown_keys():
    with pytest.raises(ValueError) as exc:
        _augment_config({"pitch_shift_prob": 1.0, "swing": 0.5})
    assert "swing" in str(exc.value)
    assert "pitch_shift_prob" not in str(exc.value)


def test_augment_config_fills_defaults_and_applies_overrides():
    assert _augment_config(None) == DEFAULT_AUGMENT
    cfg = _augment_config({"gain_db": 3.0})
    assert cfg["gain_db"] == 3.0
    assert cfg["freq_mask_width"] == DEFAULT_AUGMENT["freq_mask_width"]


def test_pitch_shift_moves_the_targets_by_the_same_semitones_as_the_audio(tmp_path, monkeypatch):
    """A sign error here trains on wrong labels with every shape unchanged."""
    import lyre.transcriber.dataset as dataset_module

    entry = {
        "audio": _write_wav(tmp_path / "a.wav", 3.0),
        "midi": _write_sustained_midi(tmp_path / "a.mid", pitch=60, seconds=3.0),
    }
    recorded = []

    def _spy(seg, sample_rate, n_steps):
        recorded.append(n_steps)
        return seg  # the shift itself is torchaudio's; we test the bookkeeping

    monkeypatch.setattr(dataset_module.AF, "pitch_shift", _spy)

    dataset = _tiny_dataset(
        [entry],
        train=True,
        augment=dict(NO_AUGMENT, enabled=True, pitch_shift_prob=1.0, pitch_shift_steps=4),
    )

    seen = set()
    for seed in range(12):
        recorded.clear()
        torch.manual_seed(seed)
        _, pitch, _, velocity = dataset[0]

        n_steps = recorded[0] if recorded else 0
        seen.add(n_steps)
        active = np.flatnonzero(pitch.numpy().max(axis=0)).tolist()
        assert active == [60 + n_steps], (
            "audio shifted by %d semitones but the pitch target sits at %r"
            % (n_steps, active)
        )
        # velocity rides the same transposition
        assert np.flatnonzero(velocity.numpy().max(axis=0)).tolist() == [60 + n_steps]

    assert seen - {0}, "no non-zero pitch shift was ever drawn"
    assert {s for s in seen if s > 0} and {s for s in seen if s < 0}


def test_augmentation_is_off_in_eval_mode(tmp_path, monkeypatch):
    import lyre.transcriber.dataset as dataset_module

    entry = {
        "audio": _write_wav(tmp_path / "a.wav", 3.0),
        "midi": _write_sustained_midi(tmp_path / "a.mid", pitch=60, seconds=3.0),
    }

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("pitch_shift ran with train=False")

    monkeypatch.setattr(dataset_module.AF, "pitch_shift", _boom)
    dataset = _tiny_dataset(
        [entry],
        train=False,
        augment=dict(NO_AUGMENT, enabled=True, pitch_shift_prob=1.0, pitch_shift_steps=4),
    )
    _, pitch, _, _ = dataset[0]
    assert np.flatnonzero(pitch.numpy().max(axis=0)).tolist() == [60]


# --------------------------------------------------------------------------
# dataset: LRU cache
# --------------------------------------------------------------------------


def _fake_item(n):
    """A (waveform, pitch, onset, velocity) tuple of a known byte size."""
    return (
        torch.zeros(n, dtype=torch.float32),
        np.zeros((n, 1), dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros((n, 1), dtype=np.float32),
    )


ITEM_BYTES = 16  # 4 float32 values per _fake_item(1) across the four arrays


def test_cache_rejects_an_item_larger_than_the_whole_budget():
    dataset = _tiny_dataset([], cache_bytes=ITEM_BYTES * 4)
    dataset._cache_put("big", _fake_item(100))
    assert dataset._cache_bytes == 0
    assert len(dataset._cache) == 0


def test_cache_bytes_track_the_retained_items():
    dataset = _tiny_dataset([], cache_bytes=ITEM_BYTES * 10)
    dataset._cache_put("a", _fake_item(2))
    assert dataset._cache_bytes == 2 * ITEM_BYTES
    dataset._cache_put("b", _fake_item(3))
    assert dataset._cache_bytes == 5 * ITEM_BYTES
    assert dataset._cache_bytes == sum(n for _, n in dataset._cache.values())


def test_reinserting_the_same_key_does_not_double_count():
    dataset = _tiny_dataset([], cache_bytes=ITEM_BYTES * 100)
    for _ in range(5):
        dataset._cache_put("a", _fake_item(4))
    assert len(dataset._cache) == 1
    assert dataset._cache_bytes == 4 * ITEM_BYTES

    # A re-insert of a different size re-accounts rather than accumulating.
    dataset._cache_put("a", _fake_item(7))
    assert dataset._cache_bytes == 7 * ITEM_BYTES


def test_cache_eviction_respects_the_budget():
    dataset = _tiny_dataset([], cache_bytes=ITEM_BYTES * 5)
    for key in "abcd":
        dataset._cache_put(key, _fake_item(2))
    assert dataset._cache_bytes <= dataset.cache_bytes
    assert dataset._cache_bytes == sum(n for _, n in dataset._cache.values())
    assert list(dataset._cache) == ["c", "d"]


def test_cache_evicts_the_least_recently_used_not_the_oldest_inserted(tmp_path):
    """Exercised through _load, which is what performs the 'use'."""
    audio = _write_wav(tmp_path / "a.wav", 0.4)
    midi = _write_midi(tmp_path / "a.mid")
    entries = [{"audio": audio, "midi": midi} for _ in range(3)]

    probe = _tiny_dataset(entries)
    probe._load(0)
    item_bytes = next(iter(probe._cache.values()))[1]

    dataset = _tiny_dataset(entries, cache_bytes=int(item_bytes * 2.5))
    dataset._load(0)
    dataset._load(1)
    dataset._load(0)  # entry 0 is now the most recently USED
    assert len(dataset._cache) == 2
    dataset._load(2)

    keys = [k[0] for k in dataset._cache]
    assert keys == [0, 2], "eviction dropped %r; entry 1 was the LRU" % (keys,)
    assert dataset._cache_bytes == sum(n for _, n in dataset._cache.values())


def test_cache_hit_returns_the_identical_arrays(tmp_path):
    entry = {
        "audio": _write_wav(tmp_path / "a.wav", 0.4),
        "midi": _write_midi(tmp_path / "a.mid"),
    }
    dataset = _tiny_dataset([entry])
    first = dataset._load(0)
    second = dataset._load(0)
    assert all(a is b for a, b in zip(first, second))


# --------------------------------------------------------------------------
# dataset: shapes and feature-config plumbing
# --------------------------------------------------------------------------


def test_hop_ms_drives_the_target_frame_rate(tmp_path):
    """The target grid was hardcoded to fps=100 regardless of hop_ms."""
    entry = {
        "audio": _write_wav(tmp_path / "a.wav", 2.0),
        "midi": _write_midi(tmp_path / "a.mid"),  # notes at 0.0 s and 1.0 s
    }

    _, pitch, onset, _ = _tiny_dataset([entry], hop_ms=10)._load(0)
    assert pitch.shape == (121, 128)
    assert np.flatnonzero(onset).tolist() == [0, 100]

    _, pitch, onset, _ = _tiny_dataset([entry], hop_ms=20)._load(0)
    assert pitch.shape == (61, 128), "hop_ms=20 must halve the target frame rate"
    assert np.flatnonzero(onset).tolist() == [0, 50]

    _, pitch, onset, _ = _tiny_dataset([entry], hop_ms=5)._load(0)
    assert pitch.shape == (241, 128)
    assert np.flatnonzero(onset).tolist() == [0, 200]


def test_min_note_and_max_note_reach_the_target_rasteriser(tmp_path):
    entry = {
        "audio": _write_wav(tmp_path / "a.wav", 2.0),
        "midi": _write_midi(tmp_path / "a.mid"),  # C4 (60) and G4 (67)
    }
    _, wide, _, _ = _tiny_dataset([entry], min_note=24, max_note=95)._load(0)
    _, narrow, _, _ = _tiny_dataset([entry], min_note=61, max_note=95)._load(0)

    assert wide[:, 60].sum() == 50.0
    assert narrow[:, 60].sum() == 0.0
    assert narrow[:, 67].sum() == 20.0


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("train", [True, False])
def test_stem_dataset_pads_a_sub_window_item(tmp_path, train):
    # 0.2 s of audio is ~20 frames -- far short of the 128-frame window.
    # This used to raise RuntimeError out of F.pad.
    entry = {
        "audio": _write_wav(tmp_path / "a.wav", 0.2),
        "midi": _write_midi(tmp_path / "a.mid"),
    }
    torch.manual_seed(0)
    dataset = StemDataset(
        [entry],
        window_frames=128,
        n_mels=32,
        n_fft=512,
        f_min=30,
        f_max=SR // 2,
        sample_rate=SR,
        train=train,
        augment={"enabled": False},
    )
    assert len(dataset) == 1
    mel, pitch, onset, velocity = dataset[0]

    assert mel.shape == (128, 32)
    assert pitch.shape == (128, 128)
    assert onset.shape == (128,)
    assert velocity.shape == (128, 128)
    for tensor in (mel, pitch, onset, velocity):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    # Targets past the end of the (very short) MIDI grid are zeroed, not clamped
    # to the last frame.
    assert float(pitch[-1].sum()) == 0.0


def test_stem_dataset_is_deterministic_in_eval_mode(tmp_path):
    entry = {
        "audio": _write_wav(tmp_path / "a.wav", 2.0),
        "midi": _write_midi(tmp_path / "a.mid"),
    }
    dataset = StemDataset(
        [entry], window_frames=64, n_mels=32, n_fft=512, f_min=30, f_max=SR // 2,
        sample_rate=SR, train=False, augment={"enabled": False},
    )
    a = dataset[0]
    b = dataset[0]
    for x, y in zip(a, b):
        assert torch.equal(x, y)


# --------------------------------------------------------------------------
# dataset: unreadable entries
#
# Decoding happens inside a DataLoader worker, so an unreadable file used to
# surface as a worker traceback that killed a training run hours in. One bad
# file in a six-figure index is a broken file; a wholly unreadable index is a
# broken corpus, and only the second is fatal.
# --------------------------------------------------------------------------


def _unreadable(tmp_path, name):
    path = tmp_path / name
    path.write_text("this is definitely not a wav file\n")
    return str(path)


def _good_entry(tmp_path, stem):
    return {
        "audio": _write_wav(tmp_path / ("%s.wav" % stem), 2.0),
        "midi": _write_midi(tmp_path / ("%s.mid" % stem)),
    }


def test_stem_dataset_steps_over_an_unreadable_entry(tmp_path, capsys):
    good = _good_entry(tmp_path, "good")
    bad = {"audio": _unreadable(tmp_path, "bad.wav"), "midi": good["midi"]}
    dataset = _tiny_dataset([bad, good])

    assert dataset.skipped == 0
    item = dataset[0]
    # The next readable entry is returned in its place, not a zero tensor.
    for x, y in zip(item, dataset[1]):
        assert torch.equal(x, y)
    assert dataset.skipped == 1

    err = capsys.readouterr().err
    assert err.count("skipping unreadable dataset entry") == 1
    assert "bad.wav" in err


def test_stem_dataset_warns_once_per_offending_path(tmp_path, capsys):
    good = _good_entry(tmp_path, "good")
    bad = {"audio": _unreadable(tmp_path, "bad.wav"), "midi": good["midi"]}
    dataset = _tiny_dataset([bad, good])

    for _ in range(5):
        dataset[0]
    err = capsys.readouterr().err
    # An epoch over a large index would otherwise print the same line once per
    # sample and bury everything else in the log.
    assert err.count("skipping unreadable dataset entry") == 1
    assert dataset.skipped == 1


def test_stem_dataset_counts_each_distinct_bad_path_once(tmp_path, capsys):
    good = _good_entry(tmp_path, "good")
    entries = [
        {"audio": _unreadable(tmp_path, "bad0.wav"), "midi": good["midi"]},
        {"audio": _unreadable(tmp_path, "bad1.wav"), "midi": good["midi"]},
        good,
    ]
    dataset = _tiny_dataset(entries)
    dataset[0]
    dataset[1]
    dataset[0]
    assert dataset.skipped == 2

    err = capsys.readouterr().err
    assert err.count("skipping unreadable dataset entry") == 2
    assert "bad0.wav" in err and "bad1.wav" in err


def test_stem_dataset_raises_when_every_entry_fails(tmp_path):
    entries = [
        {"audio": _unreadable(tmp_path, "bad0.wav"), "midi": "nope.mid"},
        {"audio": _unreadable(tmp_path, "bad1.wav"), "midi": "nope.mid"},
    ]
    dataset = _tiny_dataset(entries)
    with pytest.raises(LyreError) as excinfo:
        dataset[0]
    assert "every one of the 2 entries" in str(excinfo.value)
    assert dataset.skipped == 2


def test_stem_dataset_with_no_entries_raises_index_error():
    dataset = _tiny_dataset([])
    assert len(dataset) == 0
    with pytest.raises(IndexError):
        dataset[0]


# --------------------------------------------------------------------------
# model / inference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("total", [37, 129, 331])
def test_predict_track_covers_every_frame(total):
    torch.manual_seed(0)
    model = MultiPitchNet(n_mels=32, n_notes=128, channels=(4, 8))
    logmel = torch.randn(total, 32)

    pitch, onset = predict_track(model, logmel, window_frames=16, overlap=0.5)

    assert pitch.shape == (total, 128)
    assert onset.shape == (total, 1)
    # Every value is a sigmoid mean, so it is strictly inside (0, 1). A frame the
    # windowing missed would come back as an exact 0.
    assert float(pitch.min()) > 0.0
    assert float(pitch.max()) < 1.0
    assert float(onset.min()) > 0.0
    assert not torch.any((pitch == 0).all(dim=1))


def test_predict_track_rejects_empty_input():
    model = MultiPitchNet(n_mels=32, n_notes=128, channels=(4, 8))
    with pytest.raises(ValueError):
        predict_track(model, torch.zeros(0, 32), window_frames=16)


# The default parameter count is pinned two-sided in tests/test_features.py.


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_frame_f1_zero_tolerance_is_exact_micro_f1():
    target = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    pred = np.array([[1, 1], [0, 1], [0, 1]], dtype=np.float32)
    # TP = 3, FP = 1, FN = 1 -> P = R = F1 = 0.75
    result = frame_f1(pred, target, tolerance_frames=0)
    assert result["precision"] == pytest.approx(0.75)
    assert result["recall"] == pytest.approx(0.75)
    assert result["f1"] == pytest.approx(0.75)


def test_frame_f1_perfect_and_empty_cases():
    target = np.zeros((5, 4), dtype=np.float32)
    target[1:3, 2] = 1.0
    assert frame_f1(target, target)["f1"] == pytest.approx(1.0)
    assert frame_f1(np.zeros_like(target), target)["f1"] == 0.0
    assert frame_f1(np.zeros_like(target), np.zeros_like(target))["f1"] == 0.0


def test_frame_f1_tolerance_forgives_a_one_frame_shift():
    target = np.zeros((10, 1), dtype=np.float32)
    target[4, 0] = 1.0
    pred = np.zeros((10, 1), dtype=np.float32)
    pred[5, 0] = 1.0

    assert frame_f1(pred, target, tolerance_frames=0)["f1"] == 0.0
    assert frame_f1(pred, target, tolerance_frames=1)["f1"] == pytest.approx(1.0)


def test_frame_f1_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        frame_f1(np.zeros((4, 2)), np.zeros((5, 2)))


def test_onset_f1_accepts_flat_and_column_curves():
    target = np.zeros(20, dtype=np.float32)
    target[[3, 11]] = 1.0
    pred = np.zeros(20, dtype=np.float32)
    pred[[3, 12]] = 1.0

    flat = onset_f1(pred, target, tolerance_frames=0)
    column = onset_f1(pred[:, None], target[:, None], tolerance_frames=0)
    mixed = onset_f1(pred[:, None], target, tolerance_frames=0)

    # 1 hit, 1 false positive, 1 miss -> P = R = F1 = 0.5
    assert flat["f1"] == pytest.approx(0.5)
    assert column == flat
    assert mixed == flat

    assert onset_f1(pred[:, None], target, tolerance_frames=1)["f1"] == pytest.approx(1.0)
