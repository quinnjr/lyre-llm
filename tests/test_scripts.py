"""Entry-point script helpers.

These are the pieces that decide what a training run samples, what an eval run
reports, and whether prepare-data wrote anything at all. Every failure they used
to have was silent: a metric that always read 1.0, a device string that broke the
tool on its own defaults, a mix weight that was quietly ignored, an empty index
reported as success.
"""

import ast
import csv
import inspect
import os

import pretty_midi
import pytest
import torch
import torchaudio
import yaml

from lyre.errors import LyreError
from lyre.scripts import evaluate, export, prepare_data, train
from lyre.scripts._common import device_type, resolve_device
from lyre.tracking.hmm import Note
from lyre.transcriber.index import load_index, save_index
from lyre.transcriber.model import MultiPitchNet

SR = 16000

# A model small enough that two real training epochs finish in well under a
# second on CPU, and still wide enough to exercise the strided encoder.
TINY_FEATURES = {
    "n_mels": 16,
    "n_fft": 256,
    "f_min": 30,
    "f_max": SR // 2,
    "window_frames": 8,
    "hop_ms": 20,
    "min_note": 24,
    "max_note": 95,
}
TINY_CHANNELS = [4, 8]


def _note(pitch, start, end, velocity=100.0):
    return Note(pitch=pitch, start=start, end=end, velocity=velocity)


def _write_midi(path, notes):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = [
        pretty_midi.Note(velocity=100, pitch=n.pitch, start=n.start, end=n.end)
        for n in notes
    ]
    pm.instruments.append(inst)
    pm.write(str(path))
    return str(path)


def _write_wav(path, seconds=1.0):
    torch.manual_seed(0)
    samples = int(SR * seconds)
    wave = 0.1 * torch.sin(
        2 * torch.pi * 440.0 * torch.arange(samples, dtype=torch.float32) / SR
    )
    torchaudio.save(str(path), wave.unsqueeze(0), SR)
    return str(path)


# ------------------------------------------------------------------- _note_f1


def _reference_notes():
    return [
        _note(60, 0.0, 0.5),
        _note(62, 0.5, 1.0),
        _note(64, 1.0, 1.5),
        _note(67, 1.5, 2.0),
    ]


def test_note_f1_is_perfect_when_prediction_equals_ground_truth(tmp_path):
    gt = _write_midi(tmp_path / "gt.mid", _reference_notes())
    onset_only, full = evaluate._note_f1(_reference_notes(), gt)
    assert onset_only == pytest.approx(1.0)
    assert full == pytest.approx(1.0)


def test_note_f1_separates_onset_only_from_onset_plus_offset(tmp_path):
    # Same onsets and pitches, every duration doubled. Onset-only F1 cannot see
    # this; the offset-aware F1 must. Collapsing the two into one number is how
    # a transcriber that gets every note length wrong scores 1.0.
    gt = _write_midi(tmp_path / "gt.mid", _reference_notes())
    stretched = [_note(n.pitch, n.start, n.start + 2 * (n.end - n.start))
                 for n in _reference_notes()]

    onset_only, full = evaluate._note_f1(stretched, gt)
    assert onset_only == pytest.approx(1.0)
    assert full < 1.0


def test_note_f1_returns_two_values(tmp_path):
    gt = _write_midi(tmp_path / "gt.mid", _reference_notes())
    result = evaluate._note_f1(_reference_notes(), gt)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_note_f1_of_an_empty_prediction_is_zero(tmp_path):
    gt = _write_midi(tmp_path / "gt.mid", _reference_notes())
    assert evaluate._note_f1([], gt) == (0.0, 0.0)


def test_note_f1_against_empty_ground_truth_is_zero(tmp_path):
    gt = _write_midi(tmp_path / "empty.mid", [])
    assert evaluate._note_f1(_reference_notes(), gt) == (0.0, 0.0)


# ---------------------------------------------------------------- _playability


def test_playability_is_none_for_a_non_fretted_instrument():
    notes = [_note(36, 0.0, 0.1), _note(38, 0.5, 0.6)]
    assert evaluate._playability(notes, "drums", {}) is None
    assert evaluate._playability(notes, "vocals", {}) is None
    assert evaluate._playability(notes, "piano", {}) is None


def test_playability_is_none_for_an_empty_note_list():
    # None, not 0.0: "nothing to measure" and "nothing was playable" must not
    # average into the same number.
    assert evaluate._playability([], "guitar", {}) is None
    assert evaluate._playability([], "bass", {}) is None


def test_playability_is_one_for_an_open_position_e_minor_chord():
    chord = [_note(p, 0.0, 1.0) for p in (40, 47, 52, 55, 59, 64)]
    assert evaluate._playability(chord, "guitar", {}) == pytest.approx(1.0)


def test_playability_drops_below_one_when_a_voice_cannot_be_placed():
    # Seven simultaneous pitches on six strings: at least one must be dropped.
    crowded = [_note(p, 0.0, 1.0) for p in (40, 45, 47, 50, 52, 55, 59)]
    score = evaluate._playability(crowded, "guitar", {})
    assert score is not None
    assert score < 1.0


def test_playability_averages_over_simultaneities():
    notes = [_note(p, 0.0, 0.5) for p in (40, 47, 52)]
    notes += [_note(p, 1.0, 1.5) for p in (40, 45, 47, 50, 52, 55, 59)]
    score = evaluate._playability(notes, "guitar", {})
    assert 0.0 <= score < 1.0


# -------------------------------------------------------------- resolve_device


def test_resolve_device_never_returns_the_literal_auto():
    # Regression: "auto" was passed straight into torch.load(map_location=...),
    # which rejects it -- so `lyre eval` failed on its own default flags.
    for spec in ("auto", None):
        resolved = resolve_device(spec)
        assert resolved != "auto"
        assert resolved in ("cpu", "cuda")
        # It must be something torch actually accepts.
        torch.device(resolved)


def test_resolve_device_passes_an_explicit_device_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_device_type_strips_the_index():
    assert device_type("cuda:1") == "cuda"
    assert device_type("cpu") == "cpu"
    assert device_type(resolve_device("auto")) in ("cpu", "cuda")


# ----------------------------------------------------------- _weighted_sampler


def _entries(sources):
    return [{"source": s, "audio": f"{s}-{i}.wav"} for i, s in enumerate(sources)]


def test_weighted_sampler_warns_about_a_key_that_matches_nothing(capsys):
    # This is exactly how MAESTRO got 2.5x its intended share: the typo'd key was
    # ignored and the source silently kept the 1.0 fallback.
    notes = []
    entries = _entries(["slakh", "slakh", "maestro"])
    sampler = train._weighted_sampler(
        entries, {"slakh": 1.0, "maestro": 2.5, "mastero": 3.0}, seed=0, notes=notes
    )
    assert sampler is not None
    unmatched = [n for n in notes if "mastero" in n]
    assert len(unmatched) == 1
    assert "matches no entry" in unmatched[0]
    # The message must name what IS available, or it is unactionable.
    assert "maestro" in unmatched[0]
    assert "slakh" in unmatched[0]
    assert "mastero" in capsys.readouterr().err


def test_weighted_sampler_warns_about_a_source_with_no_weight():
    notes = []
    train._weighted_sampler(
        _entries(["slakh", "guitar"]), {"slakh": 1.0}, seed=0, notes=notes
    )
    assert any("guitar" in n and "fallback weight 1.0" in n for n in notes)


def test_weighted_sampler_is_silent_when_the_keys_line_up():
    notes = []
    train._weighted_sampler(
        _entries(["slakh", "maestro"]), {"slakh": 1.0, "maestro": 1.0}, seed=0, notes=notes
    )
    assert notes == []


def test_weighted_sampler_without_weights_is_none():
    assert train._weighted_sampler(_entries(["slakh"]), None, seed=0) is None
    assert train._weighted_sampler(_entries(["slakh"]), {}, seed=0) is None


def test_weighted_sampler_is_deterministic_for_a_fixed_seed():
    entries = _entries(["slakh"] * 4 + ["maestro"] * 2)
    weights = {"slakh": 1.0, "maestro": 2.0}
    first = list(train._weighted_sampler(entries, weights, seed=7))
    second = list(train._weighted_sampler(entries, weights, seed=7))
    assert first == second
    assert len(first) == len(entries)


# ------------------------------------------------------------- prepare_data


def _config_file(tmp_path, config, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_prepare_data_rejects_a_config_with_no_sources(tmp_path):
    path = _config_file(tmp_path, {"data": {"index_dir": str(tmp_path / "index")}})
    with pytest.raises(LyreError) as exc:
        prepare_data.main(path)
    assert "data.sources" in str(exc.value)
    # It must not have written empty indexes and called that a success.
    assert not (tmp_path / "index").exists()


def test_prepare_data_rejects_an_empty_sources_mapping(tmp_path):
    path = _config_file(tmp_path, {"data": {"sources": {}}})
    with pytest.raises(LyreError):
        prepare_data.main(path)


def test_prepare_data_rejects_a_source_with_a_null_path(tmp_path):
    path = _config_file(tmp_path, {"data": {"sources": {"maestro": None}}})
    with pytest.raises(LyreError) as exc:
        prepare_data.main(path)
    assert "maestro" in str(exc.value)


def test_prepare_data_raises_when_discovery_finds_nothing(tmp_path):
    empty = tmp_path / "corpus"
    empty.mkdir()
    index_dir = tmp_path / "index"
    path = _config_file(
        tmp_path,
        {"data": {"sources": {"maestro": str(empty)}, "index_dir": str(index_dir)}},
    )
    with pytest.raises(LyreError) as exc:
        prepare_data.main(path)
    assert "no audio/MIDI pairs" in str(exc.value)
    # Writing train.json = [] here is the silent failure: training then runs on
    # nothing and reports a loss.
    assert not index_dir.exists()


def test_prepare_data_writes_indexes_when_pairs_are_found(tmp_path):
    corpus = tmp_path / "maestro"
    corpus.mkdir()
    for i in range(6):
        (corpus / f"t{i}.wav").write_bytes(b"RIFF")
        _write_midi(corpus / f"t{i}.midi", [_note(60, 0.0, 0.5)])
    index_dir = tmp_path / "index"
    path = _config_file(
        tmp_path,
        {"data": {"sources": {"maestro": str(corpus)}, "index_dir": str(index_dir)}},
    )

    result = prepare_data.main(path)

    assert result["entries"] == 6
    assert result["failures"] == []
    for name in ("train.json", "val.json", "test.json", "maestro.json"):
        assert name in result["written"]
        assert (index_dir / name).exists()
    combined = load_index(str(index_dir / "maestro.json"))
    assert len(combined) == 6
    buckets = sum(
        len(load_index(str(index_dir / f"{b}.json"))) for b in ("train", "val", "test")
    )
    # Every entry lands in exactly one bucket -- no drops, no duplicates.
    assert buckets == 6


def test_prepare_data_honours_renamed_bucket_indexes(tmp_path):
    corpus = tmp_path / "maestro"
    corpus.mkdir()
    (corpus / "a.wav").write_bytes(b"RIFF")
    _write_midi(corpus / "a.midi", [_note(60, 0.0, 0.5)])
    index_dir = tmp_path / "index"
    path = _config_file(
        tmp_path,
        {
            "data": {
                "sources": {"maestro": str(corpus)},
                "index_dir": str(index_dir),
                "train_index": "my_train.json",
                "val_index": "my_val.json",
                "test_index": "my_test.json",
            }
        },
    )
    result = prepare_data.main(path)
    # prepare-data must write the filenames train/evaluate will read.
    assert "my_train.json" in result["written"]
    assert (index_dir / "my_train.json").exists()
    assert not (index_dir / "train.json").exists()


def test_prepare_data_reports_an_unknown_source_as_a_failure(tmp_path):
    """A typo'd source key is never intentional.

    The indexes this run writes are internally consistent but they are not the
    mixture the config describes, so the run failed. Reporting it as an advisory
    note lets `prepare-data` exit 0 on a corpus that was silently left out.
    """
    corpus = tmp_path / "maestro"
    corpus.mkdir()
    (corpus / "a.wav").write_bytes(b"RIFF")
    _write_midi(corpus / "a.midi", [_note(60, 0.0, 0.5)])
    path = _config_file(
        tmp_path,
        {
            "data": {
                "sources": {"maestro": str(corpus), "nonesuch": str(corpus)},
                "index_dir": str(tmp_path / "index"),
            }
        },
    )
    result = prepare_data.main(path)
    assert any("nonesuch" in f for f in result["failures"])
    assert result["notes"] == []
    # The known source is still indexed: one bad key is not a reason to write
    # nothing at all.
    assert result["entries"] == 1


# --------------------------------------------------------------------------
# a runnable end-to-end fixture: real audio, real MIDI, a real (tiny) model
# --------------------------------------------------------------------------


def _tiny_config(tmp_path, index_dir, **sections):
    """A complete run config that trains and evaluates in under a second."""
    config = {
        "run": {"seed": 0, "out_dir": str(tmp_path / "run")},
        "audio": {"sample_rate": SR, "decode_channels": 1},
        "features": dict(TINY_FEATURES),
        "model": {"channels": list(TINY_CHANNELS), "n_notes": 128},
        "augment": {"enabled": False},
        "data": {
            "index_dir": str(index_dir),
            "train_index": "train.json",
            "val_index": "val.json",
            "test_index": "test.json",
            "num_workers": 0,
        },
        "train": {
            "batch_size": 2,
            "epochs": 2,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "warmup_epochs": 1,
            "loss_onset_weight": 0.5,
            "grad_clip": 1.0,
            "amp": False,
            "eval_every": 1,
            "resume": None,
        },
        "tracking": {
            "p_onset": 0.05,
            "p_sustain": 0.95,
            "min_note_sec": 0.06,
            "merge_gap_sec": 0.03,
            "min_velocity": 0.2,
        },
        "arrange": {"guitar_tuning": "standard", "bass_tuning": "standard"},
        "inference": {"window_overlap": 0.5},
        "eval": {
            "frame_tolerance_frames": 2,
            "note_onset_tolerance_sec": 0.05,
            "min_instrument_f1": None,
        },
    }
    for name, values in sections.items():
        config[name] = dict(config.get(name, {}), **values)
    return config


def _tiny_entries(tmp_path, instruments=("piano", "piano", "piano", "piano"),
                  source="maestro"):
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    entries = []
    for i, instrument in enumerate(instruments):
        audio = _write_wav(media / f"t{i}.wav")
        midi = _write_midi(media / f"t{i}.mid", [_note(60 + i, 0.0, 0.5)])
        entries.append({
            "audio": audio,
            "midi": midi,
            "source": source,
            "instrument": instrument,
        })
    return entries


def _tiny_index(tmp_path, entries):
    index_dir = tmp_path / "index"
    index_dir.mkdir(exist_ok=True)
    for name in ("train.json", "val.json", "test.json"):
        save_index(str(index_dir / name), entries)
    return index_dir


def _tiny_run(tmp_path, entries=None, name="config.yaml", **sections):
    """``(config path, out dir)`` for a runnable training/eval configuration."""
    entries = _tiny_entries(tmp_path) if entries is None else entries
    index_dir = _tiny_index(tmp_path, entries)
    config = _tiny_config(tmp_path, index_dir, **sections)
    return _config_file(tmp_path, config, name), config["run"]["out_dir"]


def _tiny_model():
    return MultiPitchNet(
        n_mels=TINY_FEATURES["n_mels"], n_notes=128, channels=tuple(TINY_CHANNELS)
    )


def _write_checkpoint(path, epoch=0, step=0, best_f1=-1.0, **overrides):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    state = train._checkpoint_state(model, opt, epoch, step, best_f1)
    state.update(overrides)
    train._atomic_save(state, str(path))
    return str(path)


def _metrics_rows(out_dir):
    with open(os.path.join(out_dir, "metrics.csv"), newline="") as fh:
        return list(csv.reader(fh))


# ------------------------------------------------------------- load_checkpoint


def test_load_checkpoint_names_a_missing_file(tmp_path):
    missing = str(tmp_path / "nope.pt")
    with pytest.raises(LyreError) as exc:
        train.load_checkpoint(missing)
    assert missing in str(exc.value)


def test_load_checkpoint_rejects_a_payload_that_is_not_a_checkpoint(tmp_path):
    # A raw state_dict, a tensor, or somebody's pickled list all reach here as
    # "Missing key(s) in state_dict" from inside torch otherwise, which reads
    # like a code bug rather than "you pointed this at the wrong file".
    path = str(tmp_path / "not-a-checkpoint.pt")
    torch.save([1, 2, 3], path)
    with pytest.raises(LyreError) as exc:
        train.load_checkpoint(path)
    assert path in str(exc.value)

    torch.save({"weights": {}}, path)
    with pytest.raises(LyreError) as exc:
        train.load_checkpoint(path)
    assert path in str(exc.value)


@pytest.mark.parametrize("arch", [None, 1, 3, "2"])
def test_load_checkpoint_rejects_a_foreign_checkpoint_format(tmp_path, arch):
    path = _write_checkpoint(tmp_path / "old.pt", arch=arch)
    with pytest.raises(LyreError) as exc:
        train.load_checkpoint(path)
    message = str(exc.value)
    assert path in message
    assert str(train.CHECKPOINT_ARCH) in message


def test_load_checkpoint_accepts_what_this_build_writes(tmp_path):
    path = _write_checkpoint(tmp_path / "ok.pt", epoch=3, step=99, best_f1=0.5)
    state = train.load_checkpoint(path)
    assert state["arch"] == train.CHECKPOINT_ARCH
    assert state["epoch"] == 3
    assert state["step"] == 99
    assert state["best_f1"] == pytest.approx(0.5)
    assert set(state) >= {"model", "opt", "epoch", "step", "best_f1", "arch"}


# ----------------------------------------------------------------- _atomic_save


def test_atomic_save_removes_its_temporary_when_the_write_fails(tmp_path, monkeypatch):
    """A leftover .tmp consumes exactly the space the retry needs."""
    path = str(tmp_path / "best.pt")

    def failing_save(obj, dst, *args, **kwargs):
        with open(dst, "wb") as fh:
            fh.write(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError):
        train._atomic_save({"model": {}}, path)

    assert not os.path.exists(path + ".tmp")
    assert os.listdir(tmp_path) == []


def test_atomic_save_leaves_the_previous_checkpoint_intact_on_failure(tmp_path, monkeypatch):
    path = _write_checkpoint(tmp_path / "best.pt", epoch=1)
    before = open(path, "rb").read()

    def failing_save(obj, dst, *args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError):
        train._atomic_save({"model": {}}, path)
    assert open(path, "rb").read() == before


def test_a_saved_checkpoint_loads_under_torch_load_defaults(tmp_path):
    """torch.load defaults to weights_only=True, and the payload must survive it.

    A numpy scalar anywhere in the payload -- an ``np.float64`` learning rate
    leaking from the cosine schedule into ``opt.param_groups`` is the real case
    -- makes every checkpoint the run writes unloadable, including by this
    project's own converter.
    """
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    # Take one real step so the optimizer has moment estimates to serialise.
    sum(p.sum() for p in model.parameters()).backward()
    opt.step()

    path = str(tmp_path / "best.pt")
    train._atomic_save(train._checkpoint_state(model, opt, 1, 2, 0.25), path)

    state = torch.load(path)  # defaults, not weights_only=False
    assert state["arch"] == train.CHECKPOINT_ARCH
    assert isinstance(state["best_f1"], float)
    assert isinstance(state["epoch"], int)


def test_a_numpy_learning_rate_in_the_optimizer_would_break_that(tmp_path):
    """Negative control for the test above.

    The cosine schedule computes the LR with ``np.cos``, and the value is
    written into ``opt.param_groups`` -- from where it lands in every
    checkpoint. A numpy scalar there is not on torch.load's weights_only
    allowlist, so it makes the whole run's output unloadable, including by this
    project's own converter.
    """
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    for group in opt.param_groups:
        group["lr"] = train.np.float64(0.0005)

    path = str(tmp_path / "poisoned.pt")
    train._atomic_save(train._checkpoint_state(model, opt, 1, 2, 0.25), path)
    with pytest.raises(Exception):
        torch.load(path)
    assert torch.load(path, weights_only=False)["arch"] == train.CHECKPOINT_ARCH


# -------------------------------------------------------------------- _skip_count


@pytest.mark.parametrize(
    "value,expected",
    [(0, 0), (3, 3), (3.0, 3), ([], 0), (["a", "b"], 2), ({"a"}, 1), (None, 0)],
)
def test_skip_count_reads_whatever_shape_the_dataset_counts_in(value, expected):
    class _Dataset:
        skipped = value

    assert train._skip_count(_Dataset()) == expected


def test_skip_count_of_a_dataset_that_does_not_report_skips_is_zero():
    class _Dataset:
        pass

    assert train._skip_count(_Dataset()) == 0


# ------------------------------------------------------------------- train.main


def test_train_runs_the_configured_epochs_and_writes_checkpoints(tmp_path):
    config_path, out_dir = _tiny_run(tmp_path)
    result = train.main(config_path)

    assert result["start_epoch"] == 0
    assert result["epochs"] == 2
    assert result["failures"] == []
    assert result["skipped"] == 0
    assert os.path.exists(os.path.join(out_dir, "last.pt"))

    rows = _metrics_rows(out_dir)
    assert rows[0] == ["epoch", "step", "loss", "val_f1"]
    assert [r[0] for r in rows[1:]] == ["1", "2"]


def test_train_checkpoints_load_under_torch_load_defaults(tmp_path):
    """The end-to-end guard: whatever a real run writes must be re-loadable."""
    config_path, out_dir = _tiny_run(tmp_path)
    train.main(config_path)
    for name in ("last.pt", "best.pt"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            assert torch.load(path)["arch"] == train.CHECKPOINT_ARCH


def test_checkpoint_seeds_a_fresh_schedule_and_actually_trains(tmp_path):
    """--checkpoint is a fine-tune, not a resume: it starts at epoch 0.

    Reading `epoch` from a --checkpoint payload is how a phase-2 fine-tune of a
    60-epoch pretrain against a 30-epoch schedule trained for zero epochs and
    reported success.
    """
    seed = _write_checkpoint(tmp_path / "pretrained.pt", epoch=59, step=12000, best_f1=0.8)
    config_path, out_dir = _tiny_run(tmp_path)

    result = train.main(config_path, checkpoint=seed)

    assert result["start_epoch"] == 0
    assert result["epochs"] == 2
    assert [r[0] for r in _metrics_rows(out_dir)[1:]] == ["1", "2"]
    # The step counter is this run's, not the seed checkpoint's.
    assert int(_metrics_rows(out_dir)[1][1]) <= 4

    # And it really trained: the weights are no longer the ones it started from.
    before = train.load_checkpoint(seed)["model"]
    after = train.load_checkpoint(os.path.join(out_dir, "last.pt"))["model"]
    assert any(not torch.equal(before[k], after[k]) for k in before)

    # The best score is this run's too, not the 0.8 the seed carried.
    assert result["best_f1"] < 0.8


def test_resume_against_a_schedule_it_has_already_finished_raises(tmp_path):
    resume = _write_checkpoint(tmp_path / "last.pt", epoch=1, step=8, best_f1=0.4)
    config_path, _ = _tiny_run(tmp_path, train={"epochs": 2})
    with pytest.raises(LyreError) as exc:
        train.main(config_path, resume=resume)
    message = str(exc.value)
    assert "nothing to train" in message
    assert "--checkpoint" in message


def test_resume_one_epoch_short_of_the_end_still_trains(tmp_path):
    """The boundary the check above must not overshoot."""
    resume = _write_checkpoint(tmp_path / "last.pt", epoch=1, step=8, best_f1=-1.0)
    config_path, out_dir = _tiny_run(tmp_path, train={"epochs": 3})
    result = train.main(config_path, resume=resume)
    assert result["start_epoch"] == 2
    assert [r[0] for r in _metrics_rows(out_dir)[1:]] == ["3"]


def test_resume_restores_the_step_the_best_score_and_the_optimizer(tmp_path, monkeypatch):
    resume = _write_checkpoint(tmp_path / "last.pt", epoch=0, step=1234, best_f1=0.99)
    config_path, out_dir = _tiny_run(tmp_path, train={"epochs": 2})

    loaded = []
    original = torch.optim.AdamW.load_state_dict

    def spy(self, state):
        loaded.append(state)
        return original(self, state)

    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", spy)
    result = train.main(config_path, resume=resume)

    # Adam's moment estimates are part of the training state; dropping them
    # gives the resumed run hundreds of steps of effectively random step sizes.
    assert len(loaded) == 1

    assert result["start_epoch"] == 1
    # The LR schedule continues from the restored step rather than re-entering
    # warmup at zero.
    rows = _metrics_rows(out_dir)
    assert int(rows[1][1]) > 1234

    # A worse first evaluation after a resume must not overwrite best.pt with a
    # model the run already knows is worse than the one it saved.
    assert result["best_f1"] == pytest.approx(0.99)
    assert not os.path.exists(os.path.join(out_dir, "best.pt"))


def test_resume_appends_to_the_metrics_log_instead_of_truncating_it(tmp_path):
    config_path, out_dir = _tiny_run(tmp_path, train={"epochs": 1})
    train.main(config_path)
    first = _metrics_rows(out_dir)
    assert [r[0] for r in first[1:]] == ["1"]

    resume = os.path.join(out_dir, "last.pt")
    config_path, _ = _tiny_run(tmp_path, name="more.yaml", train={"epochs": 2})
    train.main(config_path, resume=resume)

    rows = _metrics_rows(out_dir)
    assert [r[0] for r in rows[1:]] == ["1", "2"]
    assert rows.count(["epoch", "step", "loss", "val_f1"]) == 1


def test_a_dataset_that_skipped_items_makes_the_run_a_failure(tmp_path, monkeypatch):
    """Training on less than the configured corpus is a failed run, not a note.

    The resulting model is not the one the config describes, and nothing else
    downstream can tell.
    """
    real = train.StemDataset

    class _SkippingDataset(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.skipped = 2

    monkeypatch.setattr(train, "StemDataset", _SkippingDataset)
    config_path, _ = _tiny_run(tmp_path, train={"epochs": 1})
    result = train.main(config_path)

    assert result["skipped"] == 4  # train split and val split
    assert len(result["failures"]) == 1
    assert "4 dataset item(s)" in result["failures"][0]


def test_train_reports_a_mix_weight_key_that_matches_no_source(tmp_path):
    """The end-to-end path for the typo that gave MAESTRO 2.5x its share."""
    config_path, _ = _tiny_run(
        tmp_path,
        train={"epochs": 1},
        data={"mix_weights": {"maestro": 1.0, "mastero": 3.0}},
    )
    result = train.main(config_path)
    assert any("mastero" in n and "matches no entry" in n for n in result["notes"])


# ---------------------------------------------------------------- prepare_data


def test_gate_instruments_is_the_prepare_data_vocabulary():
    """One shared list, not two that agree today.

    The eval table, the no-weak-voice gate, the playability proxy and the Demucs
    stem lookup all key on it, and a name none of them recognises is
    indistinguishable from an instrument that merely scored badly.
    """
    assert evaluate.GATE_INSTRUMENTS is prepare_data.INSTRUMENTS
    assert set(prepare_data.INSTRUMENTS) == {
        "bass", "drums", "guitar", "piano", "vocals", "other",
    }


def test_every_discovered_instrument_is_in_the_shared_vocabulary(tmp_path):
    corpus = tmp_path / "maestro"
    corpus.mkdir()
    (corpus / "a.wav").write_bytes(b"RIFF")
    _write_midi(corpus / "a.midi", [_note(60, 0.0, 0.5)])
    index_dir = tmp_path / "index"
    path = _config_file(
        tmp_path,
        {"data": {"sources": {"maestro": str(corpus)}, "index_dir": str(index_dir)}},
    )
    prepare_data.main(path)
    for entry in load_index(str(index_dir / "maestro.json")):
        assert entry["instrument"] in prepare_data.INSTRUMENTS


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Electric Bass", "bass"),
        ("Distorted Electric Guitar", "guitar"),
        ("Drums", "drums"),
        ("Piano", "piano"),
        ("Strings", "other"),
        ("Synth Pad", "other"),
        ("", "other"),
        (None, "other"),
    ],
)
def test_canonical_instrument_collapses_onto_the_vocabulary(label, expected):
    assert prepare_data.canonical_instrument(label) == expected
    assert prepare_data.canonical_instrument(label) in prepare_data.INSTRUMENTS


def _slakh_track(root, name, stems, metadata=None):
    track = root / name
    (track / "stems").mkdir(parents=True)
    (track / "MIDI").mkdir(parents=True)
    for stem_id in stems:
        (track / "stems" / f"{stem_id}.flac").write_bytes(b"fLaC")
        (track / "MIDI" / f"{stem_id}.mid").write_bytes(b"MThd")
    if metadata is not None:
        (track / "metadata.yaml").write_text(metadata)
    return track


def test_discover_slakh_reads_the_instrument_class_and_keeps_the_stem_id(tmp_path):
    """S00 says nothing about what plays on it; metadata.yaml does.

    The raw stem id is kept alongside the instrument, not instead of it: it is
    the only way back to the source file, but it is not an instrument name.
    """
    root = tmp_path / "slakh"
    _slakh_track(
        root,
        "Track00001",
        ["S00", "S01", "S02", "S03"],
        yaml.safe_dump({
            "stems": {
                "S00": {"inst_class": "Electric Bass"},
                "S01": {"inst_class": "Distorted Electric Guitar"},
                "S02": {"inst_class": "Drums"},
                "S03": {"inst_class": "Strings"},
            }
        }),
    )

    entries = {e["stem_id"]: e for e in prepare_data.discover_slakh(str(root))}

    assert set(entries) == {"S00", "S01", "S02", "S03"}
    assert entries["S00"]["instrument"] == "bass"
    assert entries["S01"]["instrument"] == "guitar"
    assert entries["S02"]["instrument"] == "drums"
    assert entries["S03"]["instrument"] == "other"
    for entry in entries.values():
        assert entry["source"] == "slakh"
        assert entry["instrument"] in prepare_data.INSTRUMENTS
        assert os.path.exists(entry["audio"])
        assert os.path.exists(entry["midi"])


@pytest.mark.parametrize(
    "metadata",
    [
        None,                                   # no metadata.yaml at all
        "stems: [not, a, mapping]\n",           # wrong shape
        "stems: {S00: {\n",                     # unparseable
        "",                                     # empty
    ],
)
def test_discover_slakh_degrades_to_other_without_aborting_the_scan(tmp_path, metadata):
    """One unreadable metadata.yaml must not end the walk of a 2100-track corpus."""
    root = tmp_path / "slakh"
    _slakh_track(root, "Track00001", ["S00"], metadata)
    _slakh_track(
        root,
        "Track00002",
        ["S00"],
        yaml.safe_dump({"stems": {"S00": {"inst_class": "Electric Bass"}}}),
    )

    entries = prepare_data.discover_slakh(str(root))

    assert len(entries) == 2
    instruments = sorted(e["instrument"] for e in entries)
    assert instruments == ["bass", "other"]


def test_discover_private_on_a_missing_root_returns_nothing(tmp_path):
    """`private` is a placeholder path in the shipped config.

    Raising here aborts the scan before the curated "check the configured
    paths" error can ever be reached.
    """
    assert prepare_data.discover_private(str(tmp_path / "nope")) == []


def test_discover_private_stamps_the_shared_other_label(tmp_path):
    root = tmp_path / "private"
    root.mkdir()
    (root / "a.wav").write_bytes(b"RIFF")
    _write_midi(root / "a.mid", [_note(60, 0.0, 0.5)])
    entries = prepare_data.discover_private(str(root))
    assert [e["instrument"] for e in entries] == ["other"]
    assert entries[0]["instrument"] in prepare_data.INSTRUMENTS


def test_a_source_root_that_does_not_exist_is_a_failure(tmp_path):
    corpus = tmp_path / "maestro"
    corpus.mkdir()
    (corpus / "a.wav").write_bytes(b"RIFF")
    _write_midi(corpus / "a.midi", [_note(60, 0.0, 0.5)])
    path = _config_file(
        tmp_path,
        {
            "data": {
                "sources": {"maestro": str(corpus), "guitar": str(tmp_path / "gone")},
                "index_dir": str(tmp_path / "index"),
            }
        },
    )
    result = prepare_data.main(path)
    # A corpus the config asks for and this machine does not have means the
    # mixture that was written is not the one the config describes.
    assert any("gone" in f for f in result["failures"])
    assert result["notes"] == []


@pytest.mark.parametrize("bucket", ["train", "val", "test"])
def test_a_bucket_index_colliding_with_a_per_source_name_raises_and_writes_nothing(
    tmp_path, bucket
):
    """The per-source pass runs second and would overwrite the mixture."""
    corpus = tmp_path / "maestro"
    corpus.mkdir()
    (corpus / "a.wav").write_bytes(b"RIFF")
    _write_midi(corpus / "a.midi", [_note(60, 0.0, 0.5)])
    index_dir = tmp_path / "index"
    path = _config_file(
        tmp_path,
        {
            "data": {
                "sources": {"maestro": str(corpus)},
                "index_dir": str(index_dir),
                f"{bucket}_index": f"maestro_{bucket}.json",
            }
        },
    )
    with pytest.raises(LyreError) as exc:
        prepare_data.main(path)
    assert f"data.{bucket}_index" in str(exc.value)
    assert not index_dir.exists()


def test_a_combined_per_source_name_also_collides(tmp_path):
    corpus = tmp_path / "maestro"
    corpus.mkdir()
    (corpus / "a.wav").write_bytes(b"RIFF")
    _write_midi(corpus / "a.midi", [_note(60, 0.0, 0.5)])
    index_dir = tmp_path / "index"
    path = _config_file(
        tmp_path,
        {
            "data": {
                "sources": {"maestro": str(corpus)},
                "index_dir": str(index_dir),
                "train_index": "maestro.json",
            }
        },
    )
    with pytest.raises(LyreError):
        prepare_data.main(path)
    assert not index_dir.exists()


def test_prepare_data_writes_each_filename_exactly_once(tmp_path):
    """A duplicate in `written` means one write silently replaced another."""
    corpora = {}
    for source, dirname in (("maestro", "maestro"), ("guitar", "GuitarSet")):
        root = tmp_path / dirname
        media = root / "audio_midi" if source == "guitar" else root
        media.mkdir(parents=True)
        for i in range(3):
            (media / f"t{i}.wav").write_bytes(b"RIFF")
            _write_midi(media / f"t{i}.midi", [_note(60, 0.0, 0.5)])
        corpora[source] = str(root)

    index_dir = tmp_path / "index"
    path = _config_file(
        tmp_path, {"data": {"sources": corpora, "index_dir": str(index_dir)}}
    )
    result = prepare_data.main(path)

    written = result["written"]
    assert len(written) == len(set(written)), "prepare-data wrote a filename twice"
    for name in written:
        assert (index_dir / name).exists()
    assert sorted(os.listdir(index_dir)) == sorted(set(written))


# -------------------------------------------------------- the no-weak-voice gate


def _scores(**by_instrument):
    return {
        name: {"frame_f1": [value] if not isinstance(value, list) else value}
        for name, value in by_instrument.items()
    }


def _all_instruments(value):
    return _scores(**{name: value for name in evaluate.GATE_INSTRUMENTS})


def test_the_gate_fails_an_instrument_below_the_threshold():
    failures, notes = [], []
    scores = _all_instruments(0.9)
    scores["drums"] = {"frame_f1": [0.10]}
    gate = evaluate._weak_voice_gate(scores, {"min_instrument_f1": 0.5}, failures, notes)

    assert gate["passed"] is False
    assert gate["weak"] == ["drums"]
    assert len(failures) == 1
    assert "drums" in failures[0]
    assert "0.500" in failures[0]


def test_the_gate_scores_every_instrument_on_its_own():
    """An average hides one unusable instrument behind five good ones."""
    failures, notes = [], []
    scores = _all_instruments(1.0)
    scores["vocals"] = {"frame_f1": [0.0]}
    gate = evaluate._weak_voice_gate(scores, {"min_instrument_f1": 0.4}, failures, notes)
    assert gate["passed"] is False
    assert gate["weak"] == ["vocals"]


def test_the_gate_fails_an_instrument_with_no_eval_tracks():
    """An un-run check is not a passed check."""
    failures, notes = [], []
    scores = _all_instruments(0.9)
    del scores["bass"]
    gate = evaluate._weak_voice_gate(scores, {"min_instrument_f1": 0.5}, failures, notes)

    assert gate["passed"] is False
    assert gate["absent"] == ["bass"]
    assert gate["scores"]["bass"] is None
    assert any("bass" in f for f in failures)


def test_the_gate_passes_when_every_instrument_clears_the_threshold():
    failures, notes = [], []
    gate = evaluate._weak_voice_gate(
        _all_instruments(0.75), {"min_instrument_f1": 0.5}, failures, notes
    )
    assert gate["passed"] is True
    assert gate["weak"] == [] and gate["absent"] == []
    assert failures == [] and notes == []
    assert set(gate["scores"]) == set(evaluate.GATE_INSTRUMENTS)
    assert gate["metric"] == evaluate.GATE_METRIC == "frame_f1"


def test_the_gate_is_reporting_only_when_no_threshold_is_configured():
    failures, notes = [], []
    scores = _all_instruments(0.9)
    del scores["guitar"]
    assert evaluate._weak_voice_gate(scores, {}, failures, notes) is None
    assert failures == []
    assert any("guitar" in n for n in notes)

    failures, notes = [], []
    assert evaluate._weak_voice_gate(
        scores, {"min_instrument_f1": None}, failures, notes
    ) is None
    assert failures == []
    assert any("guitar" in n for n in notes)


def test_the_reporting_only_gate_is_silent_when_nothing_is_missing():
    failures, notes = [], []
    assert evaluate._weak_voice_gate(
        _all_instruments(0.1), {"min_instrument_f1": None}, failures, notes
    ) is None
    assert failures == [] and notes == []


# ------------------------------------------------------------------ evaluate.main


def _eval_setup(tmp_path, entries=None, **sections):
    entries = _tiny_entries(tmp_path) if entries is None else entries
    config_path, out_dir = _tiny_run(tmp_path, entries=entries, **sections)
    checkpoint = _write_checkpoint(tmp_path / "model.pt")
    return config_path, checkpoint


def test_evaluate_does_not_count_a_passing_gate_as_a_skipped_track(tmp_path):
    """The gate appends to the same failures list the skip count is read from."""
    entries = _tiny_entries(tmp_path, instruments=evaluate.GATE_INSTRUMENTS)
    config_path, checkpoint = _eval_setup(
        tmp_path, entries=entries, eval={"min_instrument_f1": 0.0}
    )

    result = evaluate.main(config_path, checkpoint, device="cpu")

    assert result["gate"]["passed"] is True
    assert result["scored"] == len(entries)
    assert result["skipped"] == 0
    assert result["failures"] == []


def test_evaluate_counts_only_real_skips(tmp_path, monkeypatch):
    entries = _tiny_entries(tmp_path, instruments=evaluate.GATE_INSTRUMENTS)
    config_path, checkpoint = _eval_setup(
        tmp_path, entries=entries, eval={"min_instrument_f1": 0.99}
    )
    result = evaluate.main(config_path, checkpoint, device="cpu")

    # Every instrument is present and every one fails the threshold, so the gate
    # adds six messages -- none of which is a track that could not be read.
    assert result["gate"]["passed"] is False
    assert result["skipped"] == 0
    assert len(result["failures"]) == len(evaluate.GATE_INSTRUMENTS)


def test_an_out_of_memory_error_aborts_instead_of_being_logged_as_a_skip(tmp_path, monkeypatch):
    """The allocator is still saturated, so every remaining track fails the same
    way and a multi-hour eval ends with nothing but a list of skips."""
    config_path, checkpoint = _eval_setup(tmp_path)

    def oom(*args, **kwargs):
        raise MemoryError("CUDA out of memory")

    monkeypatch.setattr(evaluate, "load_mono", oom)
    with pytest.raises(MemoryError):
        evaluate.main(config_path, checkpoint, device="cpu")


def test_five_consecutive_track_failures_abort_the_run(tmp_path, monkeypatch):
    entries = _tiny_entries(tmp_path, instruments=("piano",) * 8)
    config_path, checkpoint = _eval_setup(tmp_path, entries=entries)

    def broken(*args, **kwargs):
        raise ValueError("unreadable")

    monkeypatch.setattr(evaluate, "load_mono", broken)
    with pytest.raises(LyreError) as exc:
        evaluate.main(config_path, checkpoint, device="cpu")
    message = str(exc.value)
    assert str(evaluate.MAX_CONSECUTIVE_FAILURES) in message
    assert "consecutive" in message


def test_a_run_of_failures_broken_by_a_success_does_not_abort(tmp_path, monkeypatch):
    """MAX_CONSECUTIVE_FAILURES means systemic, not cumulative."""
    entries = _tiny_entries(tmp_path, instruments=("piano",) * 8)
    config_path, checkpoint = _eval_setup(tmp_path, entries=entries)

    real = evaluate.load_mono
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 2:
            raise ValueError("unreadable")
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluate, "load_mono", flaky)
    result = evaluate.main(config_path, checkpoint, device="cpu")
    assert result["scored"] == 4
    assert result["skipped"] == 4


def test_evaluate_rejects_a_window_overlap_of_one(tmp_path):
    config_path, checkpoint = _eval_setup(tmp_path, inference={"window_overlap": 1.0})
    with pytest.raises(LyreError) as exc:
        evaluate.main(config_path, checkpoint, device="cpu")
    assert "window_overlap" in str(exc.value)


# -------------------------------------------------------------------- export.main


def _fake_onnx_export(model, example, out):
    with open(out, "wb") as fh:
        fh.write(b"onnx-graph")


def test_export_reports_one_failure_when_no_int8_backend_delivers(tmp_path, monkeypatch):
    """--int8 was asked for and nothing delivered it, so the run failed.

    The fp32 graph is real and usable, so it is still returned and named -- a
    degraded export, not nothing at all -- and no file is left under an .int8
    name that would be loaded as a quantized model.
    """
    config_path, _ = _tiny_run(tmp_path)
    checkpoint = _write_checkpoint(tmp_path / "model.pt")
    out = str(tmp_path / "model.onnx")

    def no_backend(*args, **kwargs):
        raise LyreError("no quantization backend installed")

    monkeypatch.setattr(export, "_export_onnx", _fake_onnx_export)
    monkeypatch.setattr(export, "_quantize_onnx", no_backend)
    monkeypatch.setattr(export, "_quantize_torch", no_backend)

    result = export.main(checkpoint, out, config_path, quantize=True)

    assert result["onnx"] == out
    assert os.path.exists(out)
    assert result["int8"] is None
    assert result["backend"] is None
    assert len(result["failures"]) == 1
    assert "int8" in result["failures"][0]
    assert not os.path.exists(export._int8_path(out))


def test_export_falls_back_to_the_second_backend_before_giving_up(tmp_path, monkeypatch):
    config_path, _ = _tiny_run(tmp_path)
    checkpoint = _write_checkpoint(tmp_path / "model.pt")
    out = str(tmp_path / "model.onnx")

    def no_onnxruntime(*args, **kwargs):
        raise LyreError("onnxruntime is not installed")

    def torch_backend(model, example, dst):
        with open(dst, "wb") as fh:
            fh.write(b"int8-graph")
        return "torch.ao.quantization.quantize_dynamic"

    monkeypatch.setattr(export, "_export_onnx", _fake_onnx_export)
    monkeypatch.setattr(export, "_quantize_onnx", no_onnxruntime)
    monkeypatch.setattr(export, "_quantize_torch", torch_backend)

    result = export.main(checkpoint, out, config_path, quantize=True)

    assert result["int8"] == export._int8_path(out)
    assert result["backend"] == "torch.ao.quantization.quantize_dynamic"
    assert result["failures"] == []
    # The first backend's failure is advisory once the second one worked.
    assert any("onnxruntime" in n for n in result["notes"])


def test_export_without_quantize_writes_only_the_fp32_graph(tmp_path, monkeypatch):
    config_path, _ = _tiny_run(tmp_path)
    checkpoint = _write_checkpoint(tmp_path / "model.pt")
    out = str(tmp_path / "model.onnx")
    monkeypatch.setattr(export, "_export_onnx", _fake_onnx_export)

    result = export.main(checkpoint, out, config_path)
    assert result == {
        "onnx": out, "int8": None, "backend": None, "failures": [], "notes": [],
    }
    assert not os.path.exists(export._int8_path(out))


def test_export_rejects_a_checkpoint_this_build_cannot_read(tmp_path):
    config_path, _ = _tiny_run(tmp_path)
    checkpoint = _write_checkpoint(tmp_path / "old.pt", arch=1)
    with pytest.raises(LyreError) as exc:
        export.main(checkpoint, str(tmp_path / "model.onnx"), config_path)
    assert checkpoint in str(exc.value)


# ----------------------------------------------- the script-entry-point contract


SCRIPT_MODULES = (prepare_data, train, evaluate, export)


def _module_id(module):
    return module.__name__.rsplit(".", 1)[-1]


def _own_scope(node):
    """Every node belonging to ``node``'s own scope: nested functions excluded.

    A closure's ``return`` is not the entry point's return; ``lr_at`` inside
    ``train.main`` returns a float and says nothing about main's contract.
    """
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                              ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(child))


def _returned_dicts(func):
    """Every dict literal ``func`` can return, resolving ``return <name>``."""
    body = ast.parse(inspect.getsource(func)).body[0]
    nodes = list(_own_scope(body))
    assigned = {}
    for node in nodes:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = node.value

    dicts = []
    for node in nodes:
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if isinstance(node.value, ast.Dict):
            dicts.append(node.value)
        elif isinstance(node.value, ast.Name) and node.value.id in assigned:
            dicts.append(assigned[node.value.id])
        else:
            raise AssertionError(
                "%s returns something this check cannot resolve to a dict literal"
                % func.__qualname__
            )
    return dicts


@pytest.mark.parametrize("module", SCRIPT_MODULES, ids=_module_id)
def test_every_script_main_declares_failures_and_notes_on_every_return_path(module):
    """The CLI reads `failures` and `notes` off whatever main() returns.

    A return path that omits either -- export's degraded int8 path is the one
    that nearly did -- surfaces as a KeyError in the reporting layer, or worse,
    as a run that reports no problems because nobody asked.
    """
    dicts = _returned_dicts(module.main)
    assert dicts, "%s.main returns no dict at all" % _module_id(module)
    for node in dicts:
        keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
        assert {"failures", "notes"} <= keys, (
            "a return path of %s.main omits failures/notes (has %s)"
            % (_module_id(module), sorted(keys))
        )


def _invoke_prepare_data(tmp_path, monkeypatch):
    corpus = tmp_path / "maestro"
    corpus.mkdir()
    (corpus / "a.wav").write_bytes(b"RIFF")
    _write_midi(corpus / "a.midi", [_note(60, 0.0, 0.5)])
    path = _config_file(
        tmp_path,
        {"data": {"sources": {"maestro": str(corpus)}, "index_dir": str(tmp_path / "i")}},
    )
    return prepare_data.main(path)


def _invoke_train(tmp_path, monkeypatch):
    config_path, _ = _tiny_run(tmp_path, train={"epochs": 1})
    return train.main(config_path)


def _invoke_evaluate(tmp_path, monkeypatch):
    config_path, checkpoint = _eval_setup(tmp_path)
    return evaluate.main(config_path, checkpoint, device="cpu")


def _invoke_export(tmp_path, monkeypatch):
    config_path, _ = _tiny_run(tmp_path)
    checkpoint = _write_checkpoint(tmp_path / "model.pt")
    monkeypatch.setattr(export, "_export_onnx", _fake_onnx_export)
    return export.main(checkpoint, str(tmp_path / "model.onnx"), config_path)


INVOKERS = {
    prepare_data: _invoke_prepare_data,
    train: _invoke_train,
    evaluate: _invoke_evaluate,
    export: _invoke_export,
}


@pytest.mark.parametrize("module", SCRIPT_MODULES, ids=_module_id)
def test_every_script_main_really_returns_a_failures_list(module, tmp_path, monkeypatch):
    """The declaration check above proves the shape; this proves it is reached."""
    result = INVOKERS[module](tmp_path, monkeypatch)
    assert isinstance(result, dict)
    assert isinstance(result["failures"], list)
    assert isinstance(result["notes"], list)
    for message in result["failures"] + result["notes"]:
        assert isinstance(message, str) and message


def test_every_script_module_is_covered_by_the_contract_test():
    """A fifth entry point added without a row here would be silently exempt."""
    import pkgutil

    from lyre import scripts

    modules = {
        name
        for _, name, _ in pkgutil.iter_modules(scripts.__path__)
        if not name.startswith("_")
    }
    assert modules == {_module_id(m) for m in SCRIPT_MODULES}
    assert set(INVOKERS) == set(SCRIPT_MODULES)
