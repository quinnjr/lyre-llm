from dataclasses import dataclass

import numpy as np

# The pitch axis width every stage of the system agrees on: full MIDI 0-127.
# This is the authoritative constant. Model width (``model.n_notes`` in the
# config), the rasterised targets and the decoder input must all equal it --
# a narrower model would train and export fine and then fail here, far from
# its cause, so config loading validates against this value.
N_PITCHES = 128

# The ``tracking.*`` config keys that ``frames_to_notes`` accepts. Defined here
# because ``frames_to_notes`` is what gives them meaning; callers that forward
# user config into it import this list rather than restating it.
TRACKING_KEYS = ("min_note_sec", "merge_gap_sec", "p_onset", "p_sustain", "min_velocity")


def tracking_params(tracking):
    """Select the ``frames_to_notes`` keyword arguments from a tracking config.

    Unknown keys are ignored, and absent keys fall through to the
    ``frames_to_notes`` defaults rather than being forced to None.
    """
    if not tracking:
        return {}
    return {k: tracking[k] for k in TRACKING_KEYS if k in tracking}


@dataclass
class Note:
    """A transcribed note.

    pitch:    MIDI note number, 0-127.
    start:    onset time in seconds.
    end:      offset time in seconds.
    velocity: MIDI velocity on the 0-127 scale (stored as a float, clamped to
              1-127 by the producers so a note is never silent). MIDI writers
              round and clamp this to an int in 1..127.
    """

    pitch: int
    start: float
    end: float
    velocity: float = 80.0


def _viterbi(obs, p_off_on, p_on_on):
    """Two-state (off/on) Viterbi decode.

    ``obs`` is either a 1-D activation curve of length T or a 2-D array shaped
    (T, K) holding K independent curves, which are decoded simultaneously. The
    returned states have the same shape as ``obs``.
    """
    obs = np.asarray(obs, dtype=np.float64)
    single = obs.ndim == 1
    if single:
        obs = obs[:, None]
    n, k = obs.shape

    # Observation log-likelihoods, computed once for the whole array.
    log_p_off = np.log((1.0 - obs) + 1e-12)
    log_p_on = np.log(obs + 1e-12)

    # Transition log-probabilities are loop invariant; hoist them out.
    log_off_on = np.log(p_off_on + 1e-12)
    log_off_off = np.log((1.0 - p_off_on) + 1e-12)
    log_on_on = np.log(p_on_on + 1e-12)
    log_on_off = np.log((1.0 - p_on_on) + 1e-12)

    prob_off = np.log(0.5 * (1.0 - obs[0]) + 1e-12)
    prob_on = np.log(0.5 * obs[0] + 1e-12)
    prev = np.zeros((2, n, k), dtype=np.int8)
    for t in range(1, n):
        from_off = prob_off + log_off_off
        from_on = prob_on + log_on_off
        prev[0, t] = np.where(from_off >= from_on, 0, 1)
        next_off = np.maximum(from_off, from_on) + log_p_off[t]

        from_off = prob_off + log_off_on
        from_on = prob_on + log_on_on
        prev[1, t] = np.where(from_off >= from_on, 0, 1)
        next_on = np.maximum(from_off, from_on) + log_p_on[t]

        prob_off = next_off
        prob_on = next_on

    states = np.zeros((n, k), dtype=np.int8)
    states[-1] = np.where(prob_off >= prob_on, 0, 1)
    cols = np.arange(k)
    for t in range(n - 1, 0, -1):
        states[t - 1] = prev[states[t], t, cols]
    return states[:, 0] if single else states


def _check_pitch_shape(pitch_prob, name):
    if pitch_prob.ndim != 2 or pitch_prob.shape[1] != N_PITCHES:
        raise ValueError(
            "%s expects pitch activations shaped (n_frames, %d); got %r"
            % (name, N_PITCHES, tuple(pitch_prob.shape))
        )


def viterbi_pitch_states(pitch_prob, *, p_off_on=0.05, p_on_on=0.95):
    """Decode per-pitch on/off states from a (n_frames, 128) activation map.

    Returns an int8 array shaped (n_frames, 128) with the same axis order as
    the input.

    The transition probabilities are keyword-only: this function once took a
    second positional array argument, and a stale positional call would
    otherwise bind it silently to ``p_off_on`` instead of failing.
    """
    pitch_prob = np.asarray(pitch_prob)
    _check_pitch_shape(pitch_prob, "viterbi_pitch_states")
    n_frames, n_pitches = pitch_prob.shape
    states = np.zeros((n_frames, n_pitches), dtype=np.int8)
    if n_frames == 0:
        return states
    active = np.flatnonzero(pitch_prob.max(axis=0) >= 0.3)
    if active.size:
        states[:, active] = _viterbi(pitch_prob[:, active], p_off_on, p_on_on)
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
    p_onset=0.05,
    p_sustain=0.95,
    min_velocity=0.0,
):
    """Convert frame-level activations into a list of :class:`Note`.

    ``pitch_prob`` is shaped (n_frames, 128); ``onset_prob`` is a per-frame
    scalar curve shaped (n_frames,) or (n_frames, 1). ``min_velocity`` is
    compared against the peak activation of a note on the 0-1 scale; the
    resulting :attr:`Note.velocity` is on the MIDI 0-127 scale.
    """
    pitch_prob = np.asarray(pitch_prob)
    _check_pitch_shape(pitch_prob, "frames_to_notes")
    onset = None
    if onset_prob is not None:
        onset = np.asarray(onset_prob)
        if onset.ndim > 1:
            if onset.ndim != 2 or onset.shape[1] != 1:
                raise ValueError(
                    "frames_to_notes expects onset shaped (n_frames,) or "
                    "(n_frames, 1); got %r" % (tuple(onset.shape),)
                )
            onset = onset[:, 0]
        if onset.shape[0] != pitch_prob.shape[0]:
            raise ValueError(
                "frames_to_notes: onset has %d frames but pitch has %d"
                % (onset.shape[0], pitch_prob.shape[0])
            )
    # Onsets do not influence the decode; they only refine run starts below.
    states = viterbi_pitch_states(pitch_prob, p_off_on=p_onset, p_on_on=p_sustain)
    n_frames, n_pitches = states.shape
    merge_frames = max(1, int(merge_gap_sec / frame_sec))
    min_len = max(1, int(min_note_sec / frame_sec))
    notes = []
    for p in range(n_pitches):
        starts, ends = _runs(states[:, p] == 1)
        for s, e in zip(starts, ends):
            if e - s < min_len:
                continue
            peak = float(pitch_prob[s:e, p].max())
            if peak < min_velocity:
                continue
            if onset is not None and onset_refine:
                lo = max(s - 2, 0)
                # Clamp against the run end too: the search window extends two
                # frames past `s`, so for a run shorter than that (reachable
                # with a small min_note_sec) an unclamped refinement could move
                # the start past the end and emit a negative-duration note.
                hi = min(s + 3, e, n_frames)
                if hi > lo:
                    cand = lo + int(np.argmax(onset[lo:hi]))
                    if onset[cand] > 0.5:
                        s = cand
            # The peak over the whole run, not the activation at `s`: onset
            # refinement can move `s` up to two frames before the run start,
            # and those frames are by construction below the decode threshold,
            # which would clamp nearly every note to the 1.0 floor.
            velocity = max(1.0, min(127.0, peak * 127.0))
            note = Note(
                pitch=int(p),
                start=float(s * frame_sec),
                end=float(e * frame_sec),
                velocity=velocity,
            )
            if notes and notes[-1].pitch == p and note.start - notes[-1].end <= merge_frames * frame_sec:
                notes[-1].end = note.end
            else:
                notes.append(note)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
