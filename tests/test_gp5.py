import guitarpro
import pytest

from lyre.arranger.arrange import build_arrangement
from lyre.arranger.render.gp5 import write_gp5
from lyre.instruments import Instrument
from lyre.tracking.hmm import Note


def _notes(pitches, tempo=120.0, step=0.5):
    return [
        Note(pitch=p, start=i * step, end=(i + 1) * step, velocity=0.8)
        for i, p in enumerate(pitches)
    ]


def _arrangement(ts, tuning="standard"):
    instruments = [
        Instrument(name="guitar", notes=_notes([40, 45, 50, 55, 59, 64, 52, 57])),
        Instrument(name="bass", notes=_notes([28, 33, 38, 43])),
    ]
    return build_arrangement(
        instruments,
        tempo=120.0,
        time_signature=ts,
        guitar_tuning=tuning,
        bass_tuning="standard",
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

    assert len(song.tracks) == 2
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


def test_gp5_preserves_frets_and_strings(tmp_path):
    arrangement = _arrangement((4, 4))
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


def test_gp5_empty_arrangement(tmp_path):
    arrangement = build_arrangement([], tempo=120.0, time_signature=(4, 4))
    path = tmp_path / "empty.gp5"
    write_gp5(arrangement, str(path))

    song = guitarpro.parse(str(path))
    assert song.tracks
