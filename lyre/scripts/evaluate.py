# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import os
import statistics

import mir_eval
import numpy as np
import pretty_midi
import torch

from lyre.arranger.arrange import (
    BASS_MAX_FRET,
    BASS_TUNING,
    BASS_TUNINGS,
    DEFAULT_MAX_SPAN,
    GUITAR_MAX_FRET,
    GUITAR_TUNING,
    GUITAR_TUNINGS,
    events_from_notes,
    resolve_tuning,
    voice_chord,
)
from lyre.config import load_config
from lyre.errors import LyreError
from lyre.io.decode import load_audio, load_mono
from lyre.reporting import warn
from lyre.scripts._common import resolve_device
from lyre.scripts.prepare_data import INSTRUMENTS
from lyre.scripts.train import load_checkpoint
from lyre.tracking.hmm import frames_to_notes, tracking_params
from lyre.transcriber.features import compute_features, frame_rate
from lyre.transcriber.index import load_index
from lyre.transcriber.metrics import frame_f1, onset_f1
from lyre.transcriber.model import MultiPitchNet, predict_track
from lyre.transcriber.targets import midi_to_frames

# Instruments the "no weak voice" gate covers: the design requires that no single
# instrument is allowed to lag, so the gate is per-instrument rather than an
# average. It is exactly the vocabulary prepare-data stamps on index entries --
# a gate over names no entry carries can only ever report "absent".
GATE_INSTRUMENTS = INSTRUMENTS

# A run whose every track fails is a systemic fault (a bad checkpoint, an
# exhausted allocator, a corpus that was never mounted), not per-track data rot.
# Reporting it as a long list of skips produces empty tables and a message that
# reads like a data problem.
MAX_CONSECUTIVE_FAILURES = 5

# Out-of-memory is not recoverable by moving to the next track: the allocator is
# still saturated, so every remaining track fails the same way and a multi-hour
# eval ends with nothing. These abort immediately.
OOM_ERRORS = tuple({
    e for e in (
        MemoryError,
        getattr(torch.cuda, "OutOfMemoryError", None),
        getattr(torch, "OutOfMemoryError", None),
    ) if isinstance(e, type)
})

# The metric the gate scores. Frame F1 is the one metric every instrument has,
# including drums and vocals, which have no fret-playability proxy.
GATE_METRIC = "frame_f1"

METRIC_KEYS = ("frame_f1", "onset_f1", "note_onset_f1", "note_f1", "playable")


def _to_freq(pitch):
    return 440.0 * 2 ** ((pitch - 69) / 12)


def _transcribe(model, audio, sample_rate, feats, tracking, device, overlap=0.5):
    """Transcribe a (channels, samples) waveform.

    Returns ``(pitch (T, n_notes), onset (T,), notes)``.
    """
    freq = frame_rate(sample_rate, feats["hop_ms"])
    mono = audio.mean(dim=0)
    logmel = compute_features(
        mono, sample_rate=sample_rate, n_mels=feats["n_mels"], n_fft=feats["n_fft"],
        f_min=feats["f_min"], f_max=feats["f_max"], hop_ms=feats["hop_ms"],
    )
    pitch, onset = predict_track(
        model, logmel, window_frames=feats["window_frames"],
        overlap=overlap, device=device,
    )
    notes = frames_to_notes(pitch, onset, frame_sec=1.0 / freq, **tracking_params(tracking))
    onset = np.asarray(onset)
    if onset.ndim > 1:
        onset = onset[:, 0]
    return np.asarray(pitch), onset, notes


def _ref_from_midi(path):
    pm = pretty_midi.PrettyMIDI(path)
    intervals, freqs = [], []
    for instr in pm.instruments:
        for note in instr.notes:
            intervals.append([note.start, note.end])
            freqs.append(_to_freq(note.pitch))
    if intervals:
        return np.asarray(intervals), np.asarray(freqs)
    return np.zeros((0, 2)), np.zeros(0)


def _notes_to_ref(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n.start, n.end] for n in notes])
    freqs = np.array([_to_freq(n.pitch) for n in notes])
    return intervals, freqs


def _note_f1(pred_notes, gt_midi, onset_tolerance=0.05):
    """Note-level F1 with and without the offset criterion.

    Returns ``(onset_only_f1, onset_and_offset_f1)`` -- the spec's
    "note-level onset/offset F1".
    """
    pred = _notes_to_ref(pred_notes)
    gt = _ref_from_midi(gt_midi)
    if len(pred[1]) == 0 or len(gt[1]) == 0:
        return 0.0, 0.0
    # pitch_tolerance is in CENTS; the previous 0.25 meant a quarter of a cent,
    # which no real transcription can hit. 50 cents is mir_eval's default.
    _, _, onset_only, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt[0], gt[1], pred[0], pred[1],
        onset_tolerance=onset_tolerance, pitch_tolerance=50.0, offset_ratio=None,
    )
    _, _, full, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt[0], gt[1], pred[0], pred[1],
        onset_tolerance=onset_tolerance, pitch_tolerance=50.0,
    )
    return float(onset_only), float(full)


def _playability(notes, instrument, arrange_cfg):
    """Guitar/bass tab-quality proxy: fraction of simultaneities that fret.

    Every simultaneity in ``notes`` is handed to :func:`voice_chord` (the same
    voicer the arranger uses -- nothing is reimplemented here). An event counts
    as playable only when the voicer places it with no dropped voices.
    Returns ``None`` for instruments that do not go on a fretboard.
    """
    if instrument not in ("guitar", "bass"):
        return None
    # The fretboard bounds come from the arranger's own constants: hardcoding
    # them here is how this metric came to score bass on 24 frets while the
    # arranger only ever writes 21.
    if instrument == "bass":
        tuning = resolve_tuning(arrange_cfg.get("bass_tuning"), BASS_TUNING, BASS_TUNINGS)
        max_fret = BASS_MAX_FRET
    else:
        tuning = resolve_tuning(arrange_cfg.get("guitar_tuning"), GUITAR_TUNING, GUITAR_TUNINGS)
        max_fret = GUITAR_MAX_FRET
    events = events_from_notes(notes)
    if not events:
        return None
    prev_frets = [0] * len(tuning)
    ok = 0
    for _, group in events:
        report = []
        frets = voice_chord(
            [n.pitch for n in group], tuning, prev_frets, DEFAULT_MAX_SPAN, max_fret,
            report=report,
        )
        if frets is None:
            continue
        prev_frets = [f if f >= 0 else p for f, p in zip(frets, prev_frets)]
        if not any(r.startswith("dropped") for r in report):
            ok += 1
    return ok / len(events)


def _weak_voice_gate(by_instrument, eval_cfg, failures, notes):
    """Fail the run when any single instrument scores below the threshold.

    The design calls for "no weak voice": an average hides one instrument being
    unusable, so every instrument is scored on its own. A missing instrument is a
    failure too -- an un-run check is not a passed check, and reporting it as an
    advisory is how a gate silently becomes decorative.

    Returns a dict of per-instrument results, or ``None`` when no threshold is
    configured (reporting-only mode).
    """
    threshold = eval_cfg.get("min_instrument_f1")
    if threshold is None:
        missing = [i for i in GATE_INSTRUMENTS if i not in by_instrument]
        if missing:
            warn(
                "no-weak-voice gate not run: no eval tracks for "
                + ", ".join(missing)
                + " (set eval.min_instrument_f1 to enforce a threshold)",
                notes,
            )
        return None

    threshold = float(threshold)
    scores, weak, absent = {}, [], []
    for instrument in GATE_INSTRUMENTS:
        values = (by_instrument.get(instrument) or {}).get(GATE_METRIC) or []
        if not values:
            absent.append(instrument)
            scores[instrument] = None
            continue
        score = statistics.mean(values)
        scores[instrument] = score
        if score < threshold:
            weak.append((instrument, score))

    for instrument, score in weak:
        warn(
            "no-weak-voice gate: %s %s=%.3f is below the %.3f threshold"
            % (instrument, GATE_METRIC, score, threshold),
            failures,
        )
    if absent:
        warn(
            "no-weak-voice gate: no eval tracks for "
            + ", ".join(absent)
            + "; the gate could not be applied to them",
            failures,
        )
    if not weak and not absent:
        print(
            "no-weak-voice gate: pass (all %d instruments >= %.3f %s)"
            % (len(GATE_INSTRUMENTS), threshold, GATE_METRIC)
        )
    else:
        # The verdict belongs on stdout next to the tables it judges; the
        # per-instrument detail is already on stderr with the other failures.
        print(
            "no-weak-voice gate: FAIL (%d of %d instruments did not reach %.3f %s: "
            "%d below the threshold, %d with no eval tracks)"
            % (
                len(weak) + len(absent),
                len(GATE_INSTRUMENTS),
                threshold,
                GATE_METRIC,
                len(weak),
                len(absent),
            )
        )
    return {
        "metric": GATE_METRIC,
        "threshold": threshold,
        "scores": scores,
        "weak": [i for i, _ in weak],
        "absent": absent,
        "passed": not weak and not absent,
    }


def _align(pred, target):
    n = min(pred.shape[0], target.shape[0])
    return pred[:n], target[:n]


def _record(agg, key, metrics):
    recs = agg.setdefault(key, {k: [] for k in METRIC_KEYS})
    for name, value in metrics.items():
        if value is not None:
            recs[name].append(float(value))


def _print_table(title, agg):
    print(title)
    if not agg:
        print("  (no entries)")
        return
    for key in sorted(agg):
        recs = agg[key]
        cells = []
        for name in METRIC_KEYS:
            values = recs[name]
            cells.append(f"{name}={statistics.mean(values):.3f}" if values else f"{name}=  n/a")
        print(f"  {key:10s} " + " ".join(cells) + f" n={len(recs['frame_f1'])}")


def main(config_path, checkpoint, index=None, device="auto", through_separator=False,
         separator_model="htdemucs_6s"):
    config = load_config(config_path)
    device = resolve_device(device)
    data = config["data"]
    if index is None:
        index = os.path.join(data["index_dir"], data["test_index"])
    entries = load_index(index)
    feats = config["features"]
    tracking = dict(config["tracking"])
    arrange_cfg = config.get("arrange", {})
    eval_cfg = config.get("eval", {})
    # window_overlap lives under `inference` so that transcription-time knobs are
    # in one place; evaluate must read it from there or it measures a different
    # windowing than the pipeline runs.
    overlap = float(config.get("inference", {}).get("window_overlap", 0.5))
    if not 0.0 <= overlap < 1.0:
        # At 1.0 the window step collapses to a single frame: ~128x the work,
        # which is indistinguishable from a hang.
        raise LyreError(
            f"{config_path}: inference.window_overlap is {overlap}; it must be in "
            "[0.0, 1.0). 1.0 would advance the analysis window one frame at a time."
        )
    frame_tol = int(eval_cfg.get("frame_tolerance_frames", 2))
    onset_tol_sec = float(eval_cfg.get("note_onset_tolerance_sec", 0.05))
    sample_rate = config["audio"]["sample_rate"]
    fps = frame_rate(sample_rate, feats["hop_ms"])

    model = MultiPitchNet(
        n_mels=feats["n_mels"], n_notes=config["model"]["n_notes"], channels=config["model"]["channels"]
    )
    state = load_checkpoint(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    separator = None
    if through_separator:
        from lyre.separator import Separator

        separator = Separator(model_name=separator_model, device=device)

    by_source, by_instrument = {}, {}
    failures = []
    notes_out = []
    scored = 0
    consecutive = 0
    for entry in entries:
        instrument = entry.get("instrument") or "unknown"
        # One unreadable audio file, missing MIDI or unparseable MIDI must not
        # discard the metrics of every track already scored -- an eval run is
        # hours long and prints nothing until the end.
        try:
            if separator is None:
                mono, sr = load_mono(entry["audio"], sample_rate=sample_rate)
                audio = mono.unsqueeze(0)
            else:
                waveform, sr = load_audio(
                    entry["audio"], sample_rate=sample_rate,
                    channels=config["audio"]["decode_channels"],
                )
                stems, sr = separator.separate(waveform, sr)
                if instrument in stems:
                    audio = stems[instrument]
                else:
                    # Not a Demucs stem name: measure on the re-summed separation
                    # so the degradation of the full pipeline is still captured.
                    audio = sum(stems.values())
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)

            pred_pitch, pred_onset, pred_notes = _transcribe(
                model, audio, sr, feats, tracking, device, overlap
            )
            # The training targets are filtered to features.min_note/max_note;
            # eval targets built without them would score the model on pitches it
            # was never asked to predict.
            target = midi_to_frames(
                entry["midi"], fps=fps,
                min_note=feats["min_note"], max_note=feats["max_note"],
            )
            pp, tp = _align(pred_pitch, target["pitch"])
            po, to = _align(pred_onset, target["onset"])
            note_onset, note_full = _note_f1(pred_notes, entry["midi"], onset_tol_sec)
            metrics = {
                "frame_f1": frame_f1(pp, tp, tolerance_frames=frame_tol)["f1"],
                "onset_f1": onset_f1(po, to, tolerance_frames=frame_tol)["f1"],
                "note_onset_f1": note_onset,
                "note_f1": note_full,
                "playable": _playability(pred_notes, instrument, arrange_cfg),
            }
        except OOM_ERRORS:
            raise
        except Exception as exc:
            consecutive += 1
            warn(f"skipped {entry.get('audio', '<no audio path>')}: {exc}", failures)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise LyreError(
                    f"aborting after {consecutive} consecutive track failures: this "
                    "is a fault affecting every track, not bad data in a few of "
                    f"them. Last error on {entry.get('audio', '<no audio path>')}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            continue
        consecutive = 0
        _record(by_source, entry.get("source", "unknown"), metrics)
        _record(by_instrument, instrument, metrics)
        scored += 1

    # Counted now, not from len(failures) at the end: the gate appends to the
    # same list, and a passing gate must not inflate the skip count.
    skipped = len(failures)

    route = "demucs pipeline" if through_separator else "raw stems"
    print(f"eval over {len(entries)} tracks ({index}) via {route}")
    if skipped:
        # Before the tables, so a partial eval is never read as a complete one.
        print(f"skipped {skipped}/{len(entries)} tracks (see warnings on stderr)")
    _print_table("by source:", by_source)
    _print_table("by instrument:", by_instrument)

    # After the tables: the gate's verdict is the last thing printed, and it
    # reads as a conclusion about the numbers immediately above it.
    gate = _weak_voice_gate(by_instrument, eval_cfg, failures, notes_out)
    return {
        "gate": gate,
        "by_source": by_source,
        "by_instrument": by_instrument,
        "scored": scored,
        "skipped": skipped,
        "failures": failures,
        "notes": notes_out,
    }
