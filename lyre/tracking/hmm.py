from dataclasses import dataclass

import numpy as np


@dataclass
class Note:
    pitch: int
    start: float
    end: float
    velocity: float = 80.0


def _viterbi(obs, p_off_on, p_on_on):
    n = len(obs)
    prob_off = np.zeros(n)
    prob_on = np.zeros(n)
    prev = np.zeros((2, n), dtype=np.int8)
    p_off = 1.0 - obs
    p_on = obs
    p_off_off = 1.0 - p_off_on
    p_on_off = 1.0 - p_on_on
    log_eps = np.log(1e-12)
    prob_off[0] = np.log(0.5 * p_off[0] + 1e-12)
    prob_on[0] = np.log(0.5 * p_on[0] + 1e-12)
    for t in range(1, n):
        from_off = prob_off[t - 1] + np.log(p_off_off + 1e-12)
        from_on = prob_on[t - 1] + np.log(p_on_off + 1e-12)
        prev[0, t] = 0 if from_off >= from_on else 1
        prob_off[t] = max(from_off, from_on) + np.log(p_off[t] + 1e-12)
        from_off = prob_off[t - 1] + np.log(p_off_on + 1e-12)
        from_on = prob_on[t - 1] + np.log(p_on_on + 1e-12)
        prev[1, t] = 0 if from_off >= from_on else 1
        prob_on[t] = max(from_off, from_on) + np.log(p_on[t] + 1e-12)
    states = np.zeros(n, dtype=np.int8)
    states[-1] = 0 if prob_off[-1] >= prob_on[-1] else 1
    for t in range(n - 1, 0, -1):
        states[t - 1] = prev[states[t], t]
    return states


def viterbi_pitch_states(pitch_prob, onset_prob=None, p_off_on=0.05, p_on_on=0.95):
    pitch_prob = np.asarray(pitch_prob)
    n_pitches, n_frames = pitch_prob.shape
    states = np.zeros((n_pitches, n_frames), dtype=np.int8)
    for p in range(n_pitches):
        row = pitch_prob[p]
        if row.max() < 0.3:
            continue
        states[p] = _viterbi(row, p_off_on, p_on_on)
    return states


def _runs(mask):
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


def frames_to_notes(
    pitch_prob,
    onset_prob=None,
    frame_sec=0.01,
    min_note_sec=0.06,
    merge_gap_sec=0.03,
    onset_refine=True,
):
    pitch_prob = np.asarray(pitch_prob)
    onset = np.asarray(onset_prob) if onset_prob is not None else None
    states = viterbi_pitch_states(pitch_prob, onset)
    n_pitches, _ = states.shape
    merge_frames = max(1, int(merge_gap_sec / frame_sec))
    min_len = max(1, int(min_note_sec / frame_sec))
    notes = []
    for p in range(n_pitches):
        starts, ends = _runs(states[p] == 1)
        for s, e in zip(starts, ends):
            if e - s < min_len:
                continue
            if onset is not None and onset_refine:
                lo = max(s - 2, 0)
                hi = min(e + 3, states.shape[1])
                if lo < hi:
                    peak = lo + int(np.argmax(onset[p, lo:hi]))
                    if onset[p, peak] > 0.5:
                        s = peak
            note = Note(pitch=p, start=s * frame_sec, end=e * frame_sec)
            if notes and notes[-1].pitch == p and note.start - notes[-1].end <= merge_frames * frame_sec:
                notes[-1].end = note.end
            else:
                notes.append(note)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
