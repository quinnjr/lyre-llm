import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from lyre.tracking.hmm import (
    N_PITCHES,
    TRACKING_KEYS,
    frames_to_notes,
    tracking_params,
    viterbi_pitch_states,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _blank(n_frames):
    # n_frames is deliberately != 128 so a transposed read cannot pass.
    return np.zeros((n_frames, N_PITCHES), dtype=np.float32)


def test_frames_to_notes_golden_single_sustained_note():
    pitch = _blank(250)
    pitch[50:150, 60] = 0.9
    onset = np.zeros(250, dtype=np.float32)
    onset[50] = 0.9

    notes = frames_to_notes(pitch, onset, frame_sec=0.01)

    assert len(notes) == 1
    note = notes[0]
    assert note.pitch == 60
    assert note.start == pytest.approx(0.50)
    assert note.end == pytest.approx(1.50)
    # Note.velocity is on the MIDI 0-127 scale: 0.9 * 127 == 114.3.
    assert note.velocity == pytest.approx(114.3, abs=0.05)
    assert 1.0 <= note.velocity <= 127.0


def test_frames_to_notes_accepts_onset_column_vector():
    pitch = _blank(250)
    pitch[50:150, 60] = 0.9
    onset = np.zeros((250, 1), dtype=np.float32)
    onset[50, 0] = 0.9

    flat = frames_to_notes(pitch, onset[:, 0], frame_sec=0.01)
    column = frames_to_notes(pitch, onset, frame_sec=0.01)
    assert [(n.pitch, n.start, n.end) for n in flat] == [
        (n.pitch, n.start, n.end) for n in column
    ]


@pytest.mark.parametrize("shape", [(128, 250), (250, 88), (250,)])
def test_frames_to_notes_rejects_wrong_pitch_axis(shape):
    with pytest.raises(ValueError) as exc:
        frames_to_notes(np.zeros(shape, dtype=np.float32))
    assert "128" in str(exc.value)


def test_viterbi_pitch_states_rejects_wrong_pitch_axis():
    with pytest.raises(ValueError):
        viterbi_pitch_states(np.zeros((128, 250), dtype=np.float32))


def test_viterbi_pitch_states_returns_frames_by_pitches():
    pitch = _blank(250)
    pitch[10:40, 71] = 0.95
    states = viterbi_pitch_states(pitch)
    assert states.shape == (250, N_PITCHES)
    assert states[:, 71].sum() == 30
    assert states.sum() == 30


def test_note_starting_at_frame_zero():
    # The old onset refiner indexed onset[s - 2 : s + 3] unguarded and only
    # worked for starts >= 3.
    pitch = _blank(200)
    pitch[0:100, 55] = 0.8
    onset = np.zeros(200, dtype=np.float32)
    onset[0] = 0.9

    notes = frames_to_notes(pitch, onset, frame_sec=0.01)

    assert len(notes) == 1
    assert notes[0].pitch == 55
    assert notes[0].start == pytest.approx(0.0)
    assert notes[0].end == pytest.approx(1.0)
    assert notes[0].velocity == pytest.approx(101.6, abs=0.05)


def test_min_velocity_filters_on_peak_activation():
    pitch = _blank(200)
    pitch[10:60, 50] = 0.7

    assert len(frames_to_notes(pitch, min_velocity=0.0)) == 1
    assert len(frames_to_notes(pitch, min_velocity=0.6)) == 1
    assert frames_to_notes(pitch, min_velocity=0.8) == []


def test_p_sustain_changes_the_decode():
    # A dip below 0.5 in the middle of a sustained note: a sticky on-state
    # bridges it, a leaky one splits the note in two.
    pitch = _blank(300)
    pitch[20:120, 50] = 0.9
    pitch[60:64, 50] = 0.4

    sticky = frames_to_notes(pitch, p_sustain=0.95, merge_gap_sec=0.0)
    leaky = frames_to_notes(pitch, p_sustain=0.6, merge_gap_sec=0.0)

    assert [(n.start, n.end) for n in sticky] == [(0.2, 1.2)]
    assert [(n.start, n.end) for n in leaky] == [(0.2, 0.6), (0.64, 1.2)]


def test_merge_gap_sec_merges_short_gaps():
    pitch = _blank(200)
    pitch[10:30, 50] = 0.9
    pitch[32:60, 50] = 0.9  # 2-frame (0.02 s) gap

    merged = frames_to_notes(pitch, merge_gap_sec=0.03)
    split = frames_to_notes(pitch, merge_gap_sec=0.005)

    assert [(n.start, n.end) for n in merged] == [(0.1, 0.6)]
    assert [(n.start, n.end) for n in split] == [(0.1, 0.3), (0.32, 0.6)]


def test_min_note_sec_drops_short_runs():
    pitch = _blank(200)
    pitch[10:14, 50] = 0.9  # 0.04 s -- shorter than min_note_sec
    pitch[40:80, 50] = 0.9  # 0.40 s -- kept

    notes = frames_to_notes(pitch, min_note_sec=0.06)
    assert [(n.pitch, n.start, n.end) for n in notes] == [(50, 0.4, 0.8)]

    # Lowering the floor lets the short run through.
    assert len(frames_to_notes(pitch, min_note_sec=0.01)) == 2


def test_onset_length_must_match_pitch():
    with pytest.raises(ValueError):
        frames_to_notes(_blank(200), np.zeros(199, dtype=np.float32))


# --------------------------------------------------------------------------
# onset refinement must never invert a note
# --------------------------------------------------------------------------


def test_onset_refinement_cannot_push_the_start_past_the_end():
    """The refine window used to be onset[s - 2 : s + 3], unclamped by the run.

    A 2-frame run with a spike on the frame just past it moved `start` to the
    run's end and emitted a zero- or negative-duration note. Reachable with any
    min_note_sec below ~3 frames.
    """
    pitch = _blank(60)
    pitch[10:12, 50] = 0.99  # a 2-frame run: frames 10 and 11
    onset = np.zeros(60, dtype=np.float32)
    onset[12] = 0.99  # one frame past the end of the run

    notes = frames_to_notes(pitch, onset, frame_sec=0.01, min_note_sec=0.005)

    assert len(notes) == 1
    note = notes[0]
    assert note.pitch == 50
    assert note.end > note.start, "note has non-positive duration: %r" % (note,)
    assert note.start == pytest.approx(0.10)
    assert note.end == pytest.approx(0.12)


def test_onset_refinement_still_moves_a_start_inside_the_run():
    # The clamp must not disable refinement, only bound it.
    pitch = _blank(200)
    pitch[40:120, 61] = 0.9
    onset = np.zeros(200, dtype=np.float32)
    onset[42] = 0.99  # two frames late, inside the run

    notes = frames_to_notes(pitch, onset, frame_sec=0.01)
    assert len(notes) == 1
    assert notes[0].start == pytest.approx(0.42)
    assert notes[0].end == pytest.approx(1.20)


def test_onset_refinement_can_pull_a_start_earlier():
    pitch = _blank(200)
    pitch[40:120, 61] = 0.9
    onset = np.zeros(200, dtype=np.float32)
    onset[38] = 0.99

    notes = frames_to_notes(pitch, onset, frame_sec=0.01)
    assert notes[0].start == pytest.approx(0.38)


def test_velocity_comes_from_the_run_peak_not_the_refined_start_frame():
    """A start pulled backwards lands on a frame that is below threshold.

    The refine window reaches two frames before the run, and those frames are
    by construction sub-threshold (otherwise the decode would have included
    them). Reading the activation at the refined start therefore reported ~0
    and clamped nearly every onset-refined note to the 1.0 velocity floor,
    which is inaudible-adjacent and identical for a whisper and a stab.
    """
    pitch = _blank(250)
    pitch[50:150, 60] = 0.9
    onset = np.zeros(250, dtype=np.float32)
    onset[48] = 0.9  # two frames before the run: zero pitch activation there

    notes = frames_to_notes(pitch, onset, frame_sec=0.01)

    assert len(notes) == 1
    note = notes[0]
    assert note.start == pytest.approx(0.48)  # the refinement did move it
    assert float(pitch[48, 60]) == 0.0  # onto a frame with no activation
    assert note.velocity == pytest.approx(114.3, abs=0.05)  # 0.9 * 127
    assert note.velocity > 1.0


def test_velocity_is_unchanged_by_whether_the_onset_moved_the_start():
    # The same run, decoded with and without a refining onset, must carry the
    # same dynamic -- only the start time may differ.
    pitch = _blank(250)
    pitch[50:150, 60] = 0.75
    onset = np.zeros(250, dtype=np.float32)
    onset[48] = 0.9

    moved = frames_to_notes(pitch, onset, frame_sec=0.01)[0]
    still = frames_to_notes(pitch, None, frame_sec=0.01)[0]
    assert moved.start != still.start
    assert moved.velocity == still.velocity == pytest.approx(0.75 * 127.0, abs=0.05)


def test_onset_below_the_threshold_does_not_move_the_start():
    pitch = _blank(200)
    pitch[40:120, 61] = 0.9
    onset = np.zeros(200, dtype=np.float32)
    onset[42] = 0.4  # below the 0.5 acceptance threshold

    notes = frames_to_notes(pitch, onset, frame_sec=0.01)
    assert notes[0].start == pytest.approx(0.40)


def test_onset_refine_false_ignores_the_onset_curve():
    pitch = _blank(200)
    pitch[40:120, 61] = 0.9
    onset = np.zeros(200, dtype=np.float32)
    onset[42] = 0.99

    notes = frames_to_notes(pitch, onset, frame_sec=0.01, onset_refine=False)
    assert notes[0].start == pytest.approx(0.40)


@pytest.mark.parametrize("min_note_sec", [0.001, 0.005, 0.01, 0.02, 0.06])
def test_no_note_ever_has_a_non_positive_duration(min_note_sec):
    rng = np.random.default_rng(7)
    pitch = _blank(300)
    for p in (40, 50, 60, 61, 62):
        pitch[:, p] = (rng.random(300) > 0.5).astype(np.float32) * 0.9
    onset = rng.random(300).astype(np.float32)

    for note in frames_to_notes(pitch, onset, frame_sec=0.01, min_note_sec=min_note_sec):
        assert note.end > note.start, note


# --------------------------------------------------------------------------
# viterbi_pitch_states no longer takes the onset curve
# --------------------------------------------------------------------------


def test_viterbi_pitch_states_does_not_accept_onset_prob():
    """Onsets never influenced the decode; the parameter was dead weight."""
    params = inspect.signature(viterbi_pitch_states).parameters
    assert "onset_prob" not in params
    assert list(params) == ["pitch_prob", "p_off_on", "p_on_on"]

    pitch = _blank(60)
    pitch[10:40, 50] = 0.9
    with pytest.raises(TypeError):
        viterbi_pitch_states(pitch, onset_prob=np.zeros(60, dtype=np.float32))


def test_viterbi_pitch_states_transition_probabilities_are_keyword_only():
    # The deleted second positional argument was an onset curve. A stale
    # positional call must fail rather than bind an array -- or a number that
    # was meant as something else -- silently to p_off_on.
    pitch = _blank(60)
    pitch[10:40, 50] = 0.9
    with pytest.raises(TypeError):
        viterbi_pitch_states(pitch, 0.05)
    with pytest.raises(TypeError):
        viterbi_pitch_states(pitch, np.zeros(60, dtype=np.float32))
    # The keyword form still works and still has an effect.
    assert viterbi_pitch_states(pitch, p_off_on=0.05, p_on_on=0.95).sum() == 30


def test_the_onset_curve_does_not_change_the_decoded_states():
    pitch = _blank(120)
    pitch[10:40, 50] = 0.9
    onset = np.zeros(120, dtype=np.float32)
    onset[10] = 1.0

    without = frames_to_notes(pitch, None, frame_sec=0.01)
    with_onsets = frames_to_notes(pitch, onset, frame_sec=0.01, onset_refine=False)
    assert [(n.pitch, n.start, n.end) for n in without] == [
        (n.pitch, n.start, n.end) for n in with_onsets
    ]


# --------------------------------------------------------------------------
# tracking_params: the config-to-signature coupling that broke silently
# --------------------------------------------------------------------------


def _default_tracking():
    with open(DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)["tracking"]


def test_tracking_keys_are_exactly_the_default_config_tracking_section():
    assert set(TRACKING_KEYS) == set(_default_tracking())


def test_every_tracking_key_is_a_frames_to_notes_parameter():
    params = inspect.signature(frames_to_notes).parameters
    for key in _default_tracking():
        assert key in params, "tracking.%s has no frames_to_notes parameter" % key


def test_the_default_tracking_section_drives_frames_to_notes_end_to_end():
    tracking = _default_tracking()
    selected = tracking_params(tracking)
    assert selected == tracking

    pitch = _blank(200)
    pitch[20:100, 64] = 0.9
    onset = np.zeros(200, dtype=np.float32)
    onset[20] = 0.9

    notes = frames_to_notes(pitch, onset, frame_sec=0.01, **selected)
    assert [(n.pitch, n.start, n.end) for n in notes] == [(64, 0.2, 1.0)]


def test_tracking_params_ignores_unknown_keys_and_omits_absent_ones():
    assert tracking_params(None) == {}
    assert tracking_params({}) == {}
    assert tracking_params({"p_onset": 0.1, "nonsense": 5}) == {"p_onset": 0.1}
    # An absent key falls through to the frames_to_notes default rather than
    # being forced to None.
    assert "min_velocity" not in tracking_params({"p_onset": 0.1})


def test_tracking_params_values_actually_take_effect():
    pitch = _blank(200)
    pitch[10:60, 50] = 0.7

    strict = tracking_params({"min_velocity": 0.8, "unused": 1})
    loose = tracking_params({"min_velocity": 0.1})
    assert frames_to_notes(pitch, **strict) == []
    assert len(frames_to_notes(pitch, **loose)) == 1
