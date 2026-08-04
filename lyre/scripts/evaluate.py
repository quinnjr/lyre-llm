import os
import statistics

import mir_eval
import numpy as np
import pretty_midi
import torch
import yaml

from lyre.io.decode import load_mono
from lyre.tracking.hmm import frames_to_notes
from lyre.transcriber.features import compute_features, frame_rate
from lyre.transcriber.index import load_index
from lyre.transcriber.metrics import frame_f1
from lyre.transcriber.model import MultiPitchNet, predict_track
from lyre.transcriber.targets import midi_to_frames


def _to_freq(pitch):
    return 440.0 * 2 ** ((pitch - 69) / 12)


def _transcribe(model, audio, sample_rate, feats, tracking, device):
    freq = frame_rate(sample_rate, feats["hop_ms"])
    mono = audio.mean(dim=0)
    logmel = compute_features(
        mono, sample_rate=sample_rate, n_mels=feats["n_mels"], n_fft=feats["n_fft"],
        f_min=feats["f_min"], f_max=feats["f_max"], hop_ms=feats["hop_ms"],
    )
    pitch, onset = predict_track(
        model, logmel, window_frames=feats["window_frames"],
        overlap=feats.get("window_overlap", 0.0), device=device,
    )
    params = {k: tracking[k] for k in ("min_note_sec", "merge_gap_sec") if k in tracking}
    return np.asarray(pitch), frames_to_notes(pitch, onset, frame_sec=1.0 / freq, **params)


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


def _note_f1(pred_notes, gt_midi):
    pred = _notes_to_ref(pred_notes)
    gt = _ref_from_midi(gt_midi)
    _, _, _, f1, _, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt[0], gt[1], pred[0], pred[1], onset_tolerance=0.05, pitch_tolerance=0.25
    )
    return f1


def _notes_to_ref(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n.start, n.end] for n in notes])
    freqs = np.array([_to_freq(n.pitch) for n in notes])
    return intervals, freqs


def main(config_path, checkpoint, index=None, device="auto"):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    data = config["data"]
    if index is None:
        index = os.path.join(data["index_dir"], data["test_index"])
    entries = load_index(index)
    feats = config["features"]
    tracking = dict(config["tracking"])
    tracking.pop("window_overlap", None)
    model = MultiPitchNet(
        n_mels=feats["n_mels"], n_notes=config["model"]["n_notes"], channels=config["model"]["channels"]
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    agg = {}
    for entry in entries:
        audio, sr = load_mono(entry["audio"], sample_rate=config["audio"]["sample_rate"])
        pred_pitch, pred_notes = _transcribe(model, audio.unsqueeze(0), sr, feats, tracking, device)
        target = midi_to_frames(entry["midi"], fps=100)
        ff1 = frame_f1(pred_pitch, target["pitch"])["f1"]
        nf1 = _note_f1(pred_notes, entry["midi"])
        recs = agg.setdefault(entry["source"], {"frame_f1": [], "note_f1": []})
        recs["frame_f1"].append(ff1)
        recs["note_f1"].append(nf1)

    print(f"eval over {len(entries)} tracks ({index})")
    for src, recs in agg.items():
        print(
            f"  {src:8s} frame_f1={statistics.mean(recs['frame_f1']):.3f} "
            f"note_f1={statistics.mean(recs['note_f1']):.3f} n={len(recs['frame_f1'])}"
        )