# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import inspect

import guitarpro
import pytest

from lyre.arranger.arrange import build_arrangement
from lyre.arranger.render.gp5 import (
    PERCUSSION_CHANNEL,
    UNITS_PER_WHOLE,
    _bar_units,
    write_gp5,
)
from lyre.instruments import Instrument
from lyre.tracking.hmm import Note

from conftest import make_arrangement, make_notes

# Note.velocity is on the MIDI 0-127 scale.
GUITAR_PITCHES = [40, 45, 50, 55, 59, 64, 52, 57]
BASS_PITCHES = [28, 33, 38, 43]
STEP = 0.5  # at 120 bpm one quarter note


def _notes(pitches, step=STEP):
    return make_notes(pitches, step=step)


def _arrangement(ts, tuning="standard", drums=True):
    # Eight kick/hat/snare/hat hits at 0.5 s: two 4/4 bars of a basic beat.
    return make_arrangement(
        ts,
        drums=drums,
        step=STEP,
        guitar_pitches=GUITAR_PITCHES,
        bass_pitches=BASS_PITCHES,
        drum_count=8,
        guitar_tuning=tuning,
    )


def _measure_units(measure):
    return sum(
        UNITS_PER_WHOLE // beat.duration.value for beat in measure.voices[0].beats
    )


@pytest.mark.parametrize(
    "ts,tuning",
    [((4, 4), "standard"), ((7, 8), "standard"), ((4, 4), "drop-d")],
)
def test_gp5_round_trips(tmp_path, ts, tuning):
    arrangement = _arrangement(ts, tuning)
    path = tmp_path / "score.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))

    # guitar + bass + the drum track, which is written by default.
    assert arrangement.drums
    assert len(song.tracks) == 3
    assert song.tracks[-1].name == "Drums"
    expected_bars = max(
        len(t.bars) for t in list(arrangement.guitar) + list(arrangement.bass)
    )
    assert len(song.measureHeaders) == expected_bars
    for track in song.tracks:
        assert len(track.measures) == expected_bars
        assert [s.number for s in track.strings] == list(
            range(1, len(track.strings) + 1)
        )
        assert track.measures
        for measure in track.measures:
            assert len(measure.voices) == guitarpro.Measure.maxVoices


@pytest.mark.parametrize(
    "ts",
    [(4, 4), (7, 8), (6, 8), (3, 4)],
)
def test_gp5_every_measure_fills_the_bar(tmp_path, ts):
    """Every measure sums to exactly one bar -- the invariant whose absence let
    a 4x note-duration bug ship unnoticed."""
    arrangement = _arrangement(ts)
    path = tmp_path / "score.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    expected = _bar_units(*ts)
    assert expected == ts[0] * 64 // ts[1]

    for track in song.tracks:
        for index, measure in enumerate(track.measures):
            assert measure.voices[0].beats, (
                "%s measure %d has no beats" % (track.name, index + 1)
            )
            assert _measure_units(measure) == expected, (
                "%s measure %d sums to %d units, expected %d"
                % (track.name, index + 1, _measure_units(measure), expected)
            )


def test_gp5_half_second_note_at_120bpm_is_a_quarter(tmp_path):
    # 120 bpm -> a quarter note is exactly 0.5 s. The 4x bug wrote Duration(16).
    arrangement = build_arrangement(
        [Instrument(name="guitar", notes=_notes([40, 45, 50, 55], step=0.5))],
        tempo=120.0,
        time_signature=(4, 4),
    )
    path = tmp_path / "score.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    beats = song.tracks[0].measures[0].voices[0].beats
    assert len(beats) == 4
    for beat in beats:
        assert beat.duration.value == guitarpro.Duration.quarter == 4
        assert UNITS_PER_WHOLE // beat.duration.value == 16
        assert beat.status == guitarpro.BeatStatus.normal


def test_gp5_rests_carry_an_explicit_status(tmp_path):
    # A beat left at BeatStatus.empty is read back as zero-length and collapses
    # every other beat in the measure.
    arrangement = build_arrangement(
        [Instrument(name="guitar", notes=_notes([40, 45], step=0.5))],
        tempo=120.0,
        time_signature=(4, 4),
    )
    path = tmp_path / "score.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    statuses = [
        beat.status
        for beat in song.tracks[0].measures[0].voices[0].beats
    ]
    assert guitarpro.BeatStatus.normal in statuses
    assert guitarpro.BeatStatus.rest in statuses
    assert guitarpro.BeatStatus.empty not in statuses


def test_gp5_preserves_frets_and_strings(tmp_path):
    arrangement = _arrangement((4, 4), drums=False)
    path = tmp_path / "score.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    guitar = song.tracks[0]
    n_strings = len(guitar.strings)

    played = [
        note
        for measure in guitar.measures
        for beat in measure.voices[0].beats
        for note in beat.notes
    ]
    assert played
    for note in played:
        assert 1 <= note.string <= n_strings
        assert 0 <= note.value <= guitar.fretCount

    # string.value is that string's open pitch, so fret + open pitch must
    # reconstruct the pitches we asked the arranger to voice.
    by_number = {s.number: s.value for s in guitar.strings}
    pitches = sorted(by_number[n.string] + n.value for n in played)
    assert pitches == sorted([40, 45, 50, 55, 59, 64, 52, 57])


def test_gp5_drum_track_round_trips(tmp_path):
    arrangement = _arrangement((4, 4))
    path = tmp_path / "score.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    drums = song.tracks[-1]
    assert drums.name == "Drums"
    assert drums.isPercussionTrack
    assert drums.channel.channel == PERCUSSION_CHANNEL == 9

    values = [
        note.value
        for measure in drums.measures
        for beat in measure.voices[0].beats
        for note in beat.notes
    ]
    # GM keys: kick 36, snare 38, closed hat 42 -- in the order they were hit.
    assert values == [36, 42, 38, 42, 36, 42, 38, 42]

    for index, measure in enumerate(drums.measures):
        assert _measure_units(measure) == _bar_units(4, 4)


def test_gp5_drum_track_presence_follows_the_arrangement(tmp_path):
    # `include_drums` is gone: it had no production caller and did not do what it
    # said (the writer re-gated on `if drum_hits:` internally). Drum-track
    # presence is driven purely by arrangement.drums.
    assert "include_drums" not in inspect.signature(write_gp5).parameters

    with_drums = _arrangement((4, 4), drums=True)
    assert with_drums.drums
    path = tmp_path / "with.gp5"
    write_gp5(with_drums, str(path))
    assert [t.name for t in guitarpro.parse(str(path)).tracks] == [
        "guitar", "bass", "Drums"
    ]

    without = _arrangement((4, 4), drums=False)
    assert without.drums == []
    path = tmp_path / "without.gp5"
    write_gp5(without, str(path))
    song = guitarpro.parse(str(path))
    assert [t.name for t in song.tracks] == ["guitar", "bass"]
    assert not any(t.isPercussionTrack for t in song.tracks)


def test_gp5_no_drum_track_when_there_are_no_drum_hits(tmp_path):
    # Regression: write_gp5 once created the percussion track from the
    # include_drums flag alone, so a drum-free arrangement gained a phantom
    # "Drums" track whose measures round-tripped with zero beats.
    arrangement = _arrangement((4, 4), drums=False)
    assert arrangement.drums == []
    path = tmp_path / "score.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    assert [t.name for t in song.tracks] == ["guitar", "bass"]
    for track in song.tracks:
        for measure in track.measures:
            assert _measure_units(measure) == _bar_units(4, 4)


@pytest.mark.parametrize("ts", [(4, 4), (7, 8), (3, 4)])
def test_gp5_empty_arrangement_round_trips_a_full_bar_of_rests(tmp_path, ts):
    # Regression: the empty path emitted a measure with zero beats -- 0 units
    # against a required 64 -- the same zero-length-measure shape that collapses
    # guitarpro's reader.
    arrangement = build_arrangement([], tempo=120.0, time_signature=ts)
    assert arrangement.guitar == [] and arrangement.bass == [] and arrangement.drums == []
    path = tmp_path / "empty.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    assert len(song.tracks) == 1
    assert len(song.measureHeaders) == 1
    track = song.tracks[0]
    assert len(track.measures) == 1
    for measure in track.measures:
        beats = measure.voices[0].beats
        assert beats, "empty arrangement produced a measure with zero beats"
        assert _measure_units(measure) == _bar_units(*ts)
        for beat in beats:
            assert beat.status == guitarpro.BeatStatus.rest
            assert beat.notes == []


def test_gp5_every_beat_has_an_explicit_status_everywhere(tmp_path):
    # A beat left at the attrs default (BeatStatus.empty) reads back as
    # zero-length and collapses every beat in its measure into one -- which is
    # exactly why the 4x duration bug stayed invisible under green tests.
    arrangement = _arrangement((7, 8))
    path = tmp_path / "score.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    seen = set()
    for track in song.tracks:
        for measure in track.measures:
            for beat in measure.voices[0].beats:
                assert beat.status != guitarpro.BeatStatus.empty
                assert beat.status in (
                    guitarpro.BeatStatus.rest,
                    guitarpro.BeatStatus.normal,
                )
                seen.add(beat.status)
                if beat.status == guitarpro.BeatStatus.rest:
                    assert beat.notes == []
                else:
                    assert beat.notes
    # Both kinds actually occur, so the check is not vacuous.
    assert seen == {guitarpro.BeatStatus.rest, guitarpro.BeatStatus.normal}


@pytest.mark.parametrize(
    "step,expected_value,expected_units",
    [
        (0.5, guitarpro.Duration.quarter, 16),
        (1.0, guitarpro.Duration.half, 32),
        (0.25, guitarpro.Duration.eighth, 8),
    ],
)
def test_gp5_note_lengths_map_to_the_right_duration(
    tmp_path, step, expected_value, expected_units
):
    # 120 bpm: 0.5 s is a quarter, 1.0 s a half, 0.25 s an eighth. The 4x bug
    # turned every one of these into a 64th (Duration(16)).
    arrangement = build_arrangement(
        [Instrument(name="guitar", notes=_notes([40, 45, 50, 55], step=step))],
        tempo=120.0,
        time_signature=(4, 4),
    )
    path = tmp_path / "score.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    played = [
        beat
        for measure in song.tracks[0].measures
        for beat in measure.voices[0].beats
        if beat.status == guitarpro.BeatStatus.normal
    ]
    assert len(played) == 4
    for beat in played:
        assert beat.duration.value == expected_value
        assert UNITS_PER_WHOLE // beat.duration.value == expected_units


def test_gp5_keeps_a_drum_hit_that_rounds_onto_a_bar_line(tmp_path):
    """A hit 3e-5 s before the 4.0 s bar line rounds onto it.

    Drum starts are quantised before they are grouped into events; the bar list
    they are bucketed into has to be sized from the same quantised time. When
    the two disagreed the rounded event landed past the final bar and vanished
    from the GP5 export -- while the ASCII tab, which lays its grid out
    differently, still showed it. Two exports of one arrangement disagreed.
    """
    arrangement = build_arrangement(
        [
            Instrument(name="guitar", notes=_notes([40, 45, 50, 55, 59, 64] * 2)),
            Instrument(
                name="drums",
                notes=[
                    Note(pitch=36, start=0.0, end=0.1, velocity=100.0),
                    Note(pitch=36, start=2.0, end=2.1, velocity=100.0),
                    Note(pitch=36, start=4.0 - 3e-5, end=4.1, velocity=100.0),
                ],
            ),
        ],
        tempo=120.0,
        time_signature=(4, 4),
    )
    path = tmp_path / "late.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    drums = song.tracks[-1]
    assert drums.isPercussionTrack
    assert len(drums.measures) == 3

    values = [
        note.value
        for measure in drums.measures
        for beat in measure.voices[0].beats
        for note in beat.notes
    ]
    assert values == [36, 36, 36], "the hit on the bar line was dropped"

    per_measure = [
        sum(len(beat.notes) for beat in measure.voices[0].beats)
        for measure in drums.measures
    ]
    assert per_measure == [1, 1, 1]

    for measure in drums.measures:
        assert _measure_units(measure) == _bar_units(4, 4)
