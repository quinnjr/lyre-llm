import inspect
import random

import pytest

from lyre.arranger.arrange import (
    BASS_MAX_FRET,
    DEFAULT_MAX_SPAN,
    BASS_TUNING,
    BASS_TUNINGS,
    GUITAR_MAX_FRET,
    GUITAR_TUNING,
    GUITAR_TUNINGS,
    build_bass_track,
    build_guitar_track,
    classify_drums,
    note_to_midi,
    pitch_name,
    resolve_tuning,
    voice_chord,
)
from lyre.errors import LyreError
from lyre.instruments import Instrument, _overlap_counts, split_guitar
from lyre.tracking.hmm import Note


@pytest.mark.parametrize(
    "name,expected",
    [
        ("C4", 60),
        ("C#4", 61),
        ("Db3", 49),
        ("Eb2", 39),
        ("Bb1", 34),
        ("e2", 40),
        ("A0", 21),
        ("c-1", 0),
        ("G9", 127),
        (" C4 ", 60),
    ],
)
def test_note_to_midi(name, expected):
    assert note_to_midi(name) == expected


@pytest.mark.parametrize("name", ["H4", "", "4C", "C##4", "Cbb2", "C4b", "x"])
def test_note_to_midi_rejects_junk(name):
    with pytest.raises(ValueError):
        note_to_midi(name)


def test_pitch_name():
    assert pitch_name(36) == "C2"
    assert pitch_name(60) == "C4"
    assert pitch_name(21) == "A0"
    assert pitch_name(61) == "C#4"


def test_pitch_name_round_trips_through_note_to_midi():
    for pitch in range(0, 128):
        assert note_to_midi(pitch_name(pitch)) == pitch


def test_resolve_tuning_preset():
    assert resolve_tuning("drop-d", GUITAR_TUNING, GUITAR_TUNINGS) == [38, 45, 50, 55, 59, 64]
    # normalisation: case and spaces
    assert resolve_tuning("Drop D", GUITAR_TUNING, GUITAR_TUNINGS) == [38, 45, 50, 55, 59, 64]
    assert resolve_tuning("standard", BASS_TUNING, BASS_TUNINGS) == [28, 33, 38, 43]


def test_resolve_tuning_default_and_pitch_list():
    assert resolve_tuning(None, GUITAR_TUNING, GUITAR_TUNINGS) == GUITAR_TUNING
    assert resolve_tuning([40, 45, 50, 55], GUITAR_TUNING, GUITAR_TUNINGS) == [40, 45, 50, 55]


def test_resolve_tuning_note_names():
    spec = ["Eb2", "Ab2", "Db3", "Gb3", "Bb3", "Eb4"]
    assert resolve_tuning(spec, GUITAR_TUNING, GUITAR_TUNINGS) == [39, 44, 49, 54, 58, 63]


def test_resolve_tuning_unknown_preset():
    # A bad tuning is user input, so it must surface as a LyreError the CLI can
    # report cleanly -- not a bare ValueError that escapes as a traceback.
    with pytest.raises(LyreError) as excinfo:
        resolve_tuning("nashville", GUITAR_TUNING, GUITAR_TUNINGS)
    assert "nashville" in str(excinfo.value)
    # LyreError deliberately derives from RuntimeError, not ValueError.
    assert not isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value, RuntimeError)


@pytest.mark.parametrize("bad", ["Q9", "H4", "", "4C", "C##4"])
def test_resolve_tuning_rejects_a_bad_note_name_in_a_custom_list(bad):
    # A junk note name inside an explicit tuning list is the same class of user
    # error as an unknown preset, so it gets the same exception type. It used to
    # leak note_to_midi's bare ValueError, so `["E2", "Q9"]` was a traceback
    # while `"drop-q"` was a clean message -- one function, two contracts.
    with pytest.raises(LyreError) as excinfo:
        resolve_tuning(["E2", bad], GUITAR_TUNING, GUITAR_TUNINGS)
    assert not isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value, RuntimeError)
    assert bad in str(excinfo.value) or "unknown tuning" in str(excinfo.value)


def test_resolve_tuning_still_accepts_a_mixed_list_of_names_and_numbers():
    assert resolve_tuning(["E2", 45, "D3"], GUITAR_TUNING, GUITAR_TUNINGS) == [40, 45, 50]


def test_fret_constants():
    # A 4-string bass has 21 frets; this silently drifted to 24 once.
    assert BASS_MAX_FRET == 21
    assert GUITAR_MAX_FRET == 24
    assert BASS_MAX_FRET < GUITAR_MAX_FRET


def test_build_bass_track_defaults_to_the_bass_fret_count():
    assert inspect.signature(build_bass_track).parameters["max_fret"].default == 21
    track = build_bass_track([Note(pitch=28, start=0.0, end=0.5, velocity=100.0)], 120.0)
    assert track.max_fret == BASS_MAX_FRET == 21


def test_build_guitar_track_defaults_to_the_guitar_fret_count():
    assert inspect.signature(build_guitar_track).parameters["max_fret"].default == 24
    track = build_guitar_track([Note(pitch=40, start=0.0, end=0.5, velocity=100.0)], 120.0)
    assert track.max_fret == GUITAR_MAX_FRET == 24


def test_voice_chord_puts_the_melody_on_the_top_string():
    frets = voice_chord([40, 64], GUITAR_TUNING)
    assert frets == [0, -1, -1, -1, -1, 0]
    top = max(i for i, f in enumerate(frets) if f >= 0)
    assert top == len(GUITAR_TUNING) - 1
    assert GUITAR_TUNING[top] + frets[top] == 64


def test_default_max_span_is_a_hand_span():
    # max_span is the playability constraint: it is the number of frets a hand
    # covers without shifting position. Every other test passes max_span=
    # explicitly, so nothing exercised the module default -- widening it to 9
    # left the suite green while producing tabs nobody can play.
    assert DEFAULT_MAX_SPAN == 4

    # E2 with C#5 on top: reachable only by stretching from fret 0 to fret 9.
    pitches = [40, 73]
    default = voice_chord(pitches, GUITAR_TUNING)
    assert default == voice_chord(pitches, GUITAR_TUNING, max_span=4)

    # With a four-fret hand the low voice is dropped rather than stretched to.
    assert default == [-1, -1, -1, -1, -1, 9]
    fretted = [f for f in default if f > 0]
    assert max(fretted) - min(fretted) <= 4

    # A wider hand would keep it -- which is what makes the assertion above
    # non-vacuous, and what a drifted default would silently start doing.
    assert voice_chord(pitches, GUITAR_TUNING, max_span=9) == [0, -1, -1, -1, -1, 9]


def test_voice_chord_default_max_span_bounds_every_stretch():
    # Across a sweep of two-note voicings, the default never asks for a stretch
    # wider than DEFAULT_MAX_SPAN.
    spans = []
    for low in range(40, 60):
        for high in range(low + 1, 80):
            frets = voice_chord([low, high], GUITAR_TUNING)
            fretted = [f for f in frets if f > 0]
            if len(fretted) > 1:
                spans.append(max(fretted) - min(fretted))
    assert spans, "no multi-voice fretting produced; the sweep proves nothing"
    assert max(spans) <= 4


def test_voice_chord_respects_max_span():
    # E2 is only reachable open (fret 0); D4 no lower than fret 3. Span 3.
    assert voice_chord([40, 62], GUITAR_TUNING, max_span=4) == [0, -1, -1, -1, 3, -1]
    # With max_span=2 the pair is unplayable, so the lower voice is dropped and
    # the melody (D4) survives.
    reduced = voice_chord([40, 62], GUITAR_TUNING, max_span=2)
    assert reduced == [-1, -1, -1, -1, 3, -1]


def test_voice_chord_respects_max_fret():
    # E6 sits at fret 24 of the top string.
    assert voice_chord([88], GUITAR_TUNING, max_fret=24) == [-1, -1, -1, -1, -1, 24]
    # A 12-fret neck cannot reach it, so it is dropped an octave to fret 12.
    report = []
    frets = voice_chord([88], GUITAR_TUNING, max_fret=12, report=report)
    assert frets == [-1, -1, -1, -1, -1, 12]
    assert report == ["shifted E6 down 1 octave"]


def test_voice_chord_octave_shifts_and_reports():
    report = []
    frets = voice_chord([20], GUITAR_TUNING, report=report)
    assert frets == [4, -1, -1, -1, -1, -1]
    assert GUITAR_TUNING[0] + frets[0] == 44  # G#0 raised two octaves to G#2
    assert report == ["shifted G#0 up 2 octaves"]


def test_voice_chord_reduces_polyphony_and_reports_the_drop():
    report = []
    # Seven distinct pitches on a six-string instrument.
    frets = voice_chord([40, 45, 50, 55, 59, 64, 67], GUITAR_TUNING, report=report)
    assert sum(1 for f in frets if f >= 0) <= len(GUITAR_TUNING)
    assert any("dropped" in line for line in report)
    # The melody (highest pitch) survives the reduction.
    sounded = [GUITAR_TUNING[i] + f for i, f in enumerate(frets) if f >= 0]
    assert max(sounded) == 67


def test_voice_chord_does_not_report_a_drop_for_an_octave_doubled_power_chord():
    # E2 / B2 / E3 -- a power chord with the root doubled at the octave. All
    # three voices are playable, so the arranger must stay silent. The
    # simultaneity count used to be taken before octave folding/de-duplication,
    # so every power chord produced a spurious "dropped 1 of 3" note.
    report = []
    frets = voice_chord([40, 52, 47], GUITAR_TUNING, report=report)
    assert frets is not None
    sounded = sorted(GUITAR_TUNING[i] + f for i, f in enumerate(frets) if f >= 0)
    assert sounded == [40, 47, 52]
    assert report == []


def test_voice_chord_does_not_report_a_drop_when_octave_folding_collapses_a_voice():
    # E1 is below the guitar's range, so it is folded up an octave and lands
    # exactly on the E2 that is already in the chord. That is one voice, not a
    # dropped one -- the simultaneity count must be taken AFTER folding.
    report = []
    frets = voice_chord([28, 40, 47], GUITAR_TUNING, report=report)
    sounded = sorted(GUITAR_TUNING[i] + f for i, f in enumerate(frets) if f >= 0)
    assert sounded == [40, 47]
    assert report == ["shifted E1 up 1 octave"]
    assert not any("dropped" in line for line in report)


def test_voice_chord_does_not_report_a_drop_for_a_unison_doubled_chord():
    # The same pitch requested twice is one voice, not a dropped one.
    report = []
    frets = voice_chord([40, 40, 47], GUITAR_TUNING, report=report)
    sounded = sorted(GUITAR_TUNING[i] + f for i, f in enumerate(frets) if f >= 0)
    assert sounded == [40, 47]
    assert report == []


def test_voice_chord_still_reports_a_genuine_drop():
    # Seven distinct, non-octave-equivalent pitches on six strings: a real loss.
    report = []
    frets = voice_chord([40, 45, 50, 55, 59, 64, 68], GUITAR_TUNING, report=report)
    assert sum(1 for f in frets if f >= 0) <= 6
    drops = [line for line in report if "dropped" in line]
    assert len(drops) == 1
    assert drops[0] == "dropped 1 of 7 simultaneous notes"


def test_voice_chord_empty():
    assert voice_chord([], GUITAR_TUNING) is None


# --------------------------------------------------------------------------
# _overlap_counts / split_guitar -- differential against the original O(n^2)
# --------------------------------------------------------------------------


def _overlap_oracle(notes, eps=1e-3):
    """The pre-rewrite double loop, kept verbatim as a reference implementation."""
    return [
        sum(
            1
            for other in notes
            if other is not note
            and other.end > note.start + eps
            and other.start < note.end - eps
        )
        for note in notes
    ]


def _n(start, end, pitch=60):
    return Note(pitch=pitch, start=start, end=end, velocity=100.0)


def _disjoint():
    return [_n(0.0, 1.0), _n(2.0, 3.0), _n(4.0, 5.0)]


def _stacked_chord():
    return [_n(1.0, 2.0, 40), _n(1.0, 2.0, 47), _n(1.0, 2.0, 52)]


def _touching_chain():
    # Each note overlaps only its immediate neighbours.
    return [_n(0.0, 1.5), _n(1.0, 2.5), _n(2.0, 3.5), _n(3.0, 4.5)]


def _abutting():
    # end == next start: touching but NOT overlapping, the classic off-by-one.
    return [_n(0.0, 1.0), _n(1.0, 2.0), _n(2.0, 3.0)]


def _degenerate():
    # start == end forces the `lo < hi` fallback branch, mixed with real notes.
    return [_n(0.0, 0.0), _n(0.5, 0.5), _n(0.0, 2.0), _n(0.4, 0.6), _n(1.0, 1.0)]


def _nested():
    return [_n(0.0, 10.0), _n(1.0, 2.0), _n(3.0, 4.0), _n(3.5, 9.0)]


def _random_set():
    rng = random.Random(20260804)
    notes = []
    for _ in range(120):
        start = round(rng.uniform(0.0, 20.0), 3)
        length = rng.choice([0.0, 0.0005, 0.05, 0.25, 1.0, 3.0])
        notes.append(_n(start, round(start + length, 4)))
    return notes


@pytest.mark.parametrize(
    "name,factory",
    [
        ("empty", list),
        ("single", lambda: [_n(0.0, 1.0)]),
        ("disjoint", _disjoint),
        ("stacked_chord", _stacked_chord),
        ("touching_chain", _touching_chain),
        ("abutting", _abutting),
        ("degenerate", _degenerate),
        ("nested", _nested),
        ("random", _random_set),
    ],
)
def test_overlap_counts_matches_the_quadratic_oracle(name, factory):
    notes = factory()
    assert _overlap_counts(notes) == _overlap_oracle(notes), name


def test_overlap_counts_known_values():
    # Pin the oracle itself down so a matching pair of wrong answers cannot pass.
    assert _overlap_counts(_disjoint()) == [0, 0, 0]
    assert _overlap_counts(_stacked_chord()) == [2, 2, 2]
    assert _overlap_counts(_touching_chain()) == [1, 2, 2, 1]
    assert _overlap_counts(_abutting()) == [0, 0, 0]
    assert _overlap_counts(_nested()) == [3, 1, 2, 2]


def test_overlap_counts_never_counts_a_note_against_itself():
    # One character in the `count - 1` self-exclusion separates 0 from 1.
    assert _overlap_counts([_n(0.0, 1.0)]) == [0]
    assert _overlap_counts([_n(0.0, 0.0)]) == [0]


def test_split_guitar_sends_chords_to_rhythm_and_a_line_to_lead():
    chord = _stacked_chord()
    rhythm, lead = split_guitar(Instrument(name="guitar", notes=chord))
    assert rhythm == chord
    assert lead == []

    line = [_n(i * 0.25, i * 0.25 + 0.2, 60 + i) for i in range(8)]
    rhythm, lead = split_guitar(Instrument(name="guitar", notes=line))
    assert rhythm == []
    assert lead == line


def test_split_guitar_partitions_without_loss():
    notes = _stacked_chord() + [_n(5.0, 5.5, 64), _n(6.0, 6.5, 67)]
    instrument = Instrument(name="guitar", notes=notes)
    rhythm, lead = split_guitar(instrument)
    # Every note lands on exactly one side, and source order is preserved.
    assert [id(n) for n in rhythm] == [id(n) for n in notes[:3]]
    assert [id(n) for n in lead] == [id(n) for n in notes[3:]]


def test_note_sustaining_across_a_barline_is_emitted_once():
    # A note that rings across a barline belongs to the bar it starts in, once.
    # 120 bpm 4/4 -> 2.0 s bars. The note starts in bar 1 and rings into bar 2.
    track = build_guitar_track(
        [Note(pitch=40, start=1.5, end=3.5, velocity=100.0)], 120.0, (4, 4)
    )
    assert track.bars == [(0.0, 2.0), (2.0, 3.5)]
    assert len(track.events) == 1
    event = track.events[0]
    assert event.start == pytest.approx(1.5)
    assert len(event.notes) == 1
    tab_note = event.notes[0]
    assert tab_note.pitch == 40
    assert tab_note.start == pytest.approx(1.5)
    # True end 3.5 s clipped to the onset bar's end.
    assert tab_note.end == pytest.approx(2.0)


def test_classify_drums_maps_general_midi_keys():
    notes = [
        Note(pitch=36, start=0.0, end=0.1, velocity=100.0),
        Note(pitch=38, start=0.5, end=0.6, velocity=100.0),
        Note(pitch=40, start=0.75, end=0.8, velocity=100.0),
        Note(pitch=42, start=1.0, end=1.1, velocity=100.0),
        Note(pitch=46, start=1.25, end=1.3, velocity=100.0),
        Note(pitch=49, start=1.5, end=1.6, velocity=100.0),
        Note(pitch=51, start=1.75, end=1.8, velocity=100.0),
        Note(pitch=45, start=2.0, end=2.1, velocity=100.0),
        Note(pitch=60, start=2.5, end=2.6, velocity=100.0),  # not percussion
    ]
    hits = classify_drums(notes)
    assert [(h.start, h.part) for h in hits] == [
        (0.0, "kick"),
        (0.5, "snare"),
        (0.75, "snare"),
        (1.0, "hat"),
        (1.25, "hat"),
        (1.5, "crash"),
        (1.75, "ride"),
        (2.0, "tom"),
    ]
