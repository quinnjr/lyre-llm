import inspect
import re
from xml.etree import ElementTree as ET

import pytest

from lyre.arranger.arrange import (
    Arrangement,
    build_arrangement,
    build_bass_track,
    build_guitar_track,
)
from lyre.arranger.render._common import DRUM_TIME_PLACES, drum_bars, drum_events
from lyre.arranger.render.ascii import render_all_ascii, render_parts
from lyre.arranger.render.musicxml import _bar_units, write_musicxml
from lyre.instruments import Instrument
from lyre.tracking.hmm import Note

from conftest import make_arrangement as _arrangement
from conftest import make_notes as _notes

# <type> name -> duration in divisions (DIVISIONS = 4, so a whole note is 16).
TYPE_UNITS = {"whole": 16, "half": 8, "quarter": 4, "eighth": 2, "16th": 1}

_TAB_ROW = re.compile(r"^[A-Ga-g]\|.*\|$")


# --------------------------------------------------------------------------
# ASCII
# --------------------------------------------------------------------------


def test_ascii_two_digit_fret_keeps_rows_aligned():
    # Fret 12 on the top string next to single-digit frets elsewhere.
    arrangement = build_arrangement(
        [Instrument(name="guitar", notes=_notes([64, 76, 40, 52, 45, 57, 50, 62]))],
        tempo=120.0,
        time_signature=(4, 4),
    )
    text = render_all_ascii(arrangement)
    rows = [line for line in text.splitlines() if _TAB_ROW.match(line)]

    assert len(rows) == 6
    assert "12" in "".join(rows)
    assert len({len(r) for r in rows}) == 1
    # Every row ends on the barline, so the bar separators line up too.
    assert len({r.index("|") for r in rows}) == 1


def test_ascii_prints_performance_notes_when_present():
    # G#0 is below the guitar's range; the arranger octave-shifts it and records
    # the compromise in performance_notes, which the renderer must surface.
    arrangement = build_arrangement(
        [Instrument(name="guitar", notes=_notes([20, 64, 59, 55]))],
        tempo=120.0,
        time_signature=(4, 4),
    )
    track = arrangement.guitar[0]
    assert track.performance_notes  # arranger recorded something

    text = render_all_ascii(arrangement)
    assert f"{track.name} notes:" in text
    for line in track.performance_notes:
        assert f"  - {line}" in text
    assert "shifted G#0 up 2 octaves" in text


def test_ascii_omits_the_notes_block_when_there_is_nothing_to_say():
    arrangement = build_arrangement(
        [Instrument(name="guitar", notes=_notes([40, 45, 50, 55]))],
        tempo=120.0,
        time_signature=(4, 4),
    )
    assert arrangement.guitar[0].performance_notes == []
    assert "notes:" not in render_all_ascii(arrangement)


# --------------------------------------------------------------------------
# render_parts -- the (guitar, bass, drums) bundle
# --------------------------------------------------------------------------


def _adversarial_parts():
    """Names that lie: a guitar called "Bass Guitar", a bass called "gtr 2"."""
    guitar = build_guitar_track(
        _notes([40, 45, 50, 55, 59, 64, 52, 57]), 120.0, (4, 4), name="Bass Guitar"
    )
    # Deliberately different frets from the guitar part so a row that shows up
    # in the wrong render is unambiguous rather than a coincidental collision.
    bass = build_bass_track(
        _notes([31, 34, 37, 41, 44, 47, 36, 39]), 120.0, (4, 4), name="gtr 2"
    )
    return Arrangement(tempo=120.0, ts=(4, 4), guitar=[guitar], bass=[bass]), guitar, bass


def test_render_parts_groups_by_the_arrangement_not_by_track_name():
    # Regression: the splitter matched the capitalised substrings "Guitar"/"Bass"
    # against lowercased labels, so bass.tab was written as zero bytes while
    # guitar.tab held the whole render.
    arrangement, guitar, bass = _adversarial_parts()
    guitar_text, bass_text, drums_text = render_parts(arrangement)

    assert guitar_text.strip()
    assert bass_text.strip()  # the zero-byte bug

    # Grouping follows object identity, despite the misleading names.
    assert "Bass Guitar (" in guitar_text
    assert "gtr 2 (" in bass_text
    assert "gtr 2" not in guitar_text
    assert "Bass Guitar" not in bass_text

    # Neither render is a slice of the other, and the bass tab body is absent
    # from the guitar text (the old code string-subtracted an empty marker).
    assert guitar_text not in bass_text
    assert bass_text not in guitar_text
    bass_rows = [line for line in bass_text.splitlines() if _TAB_ROW.match(line)]
    assert len(bass_rows) == 4 * len(bass.bars)  # 4 strings per bar
    for row in bass_rows:
        assert row not in guitar_text

    guitar_rows = [line for line in guitar_text.splitlines() if _TAB_ROW.match(line)]
    assert len(guitar_rows) == 6 * len(guitar.bars)

    # No drum hits in this arrangement.
    assert drums_text == "[Drums: no content]"


def test_render_parts_matches_the_single_track_renders():
    arrangement, guitar, bass = _adversarial_parts()
    guitar_text, bass_text, _ = render_parts(arrangement)

    guitar_only = Arrangement(tempo=120.0, ts=(4, 4), guitar=[guitar])
    bass_only = Arrangement(tempo=120.0, ts=(4, 4), bass=[bass])
    g_all = render_all_ascii(guitar_only)
    b_all = render_all_ascii(bass_only)
    assert guitar_text and guitar_text in g_all
    assert bass_text and bass_text in b_all


def test_render_parts_handles_empty_groups():
    arrangement, guitar, _bass = _adversarial_parts()
    only_guitar = Arrangement(tempo=120.0, ts=(4, 4), guitar=[guitar])
    guitar_text, bass_text, drums_text = render_parts(only_guitar)
    assert guitar_text.strip()
    assert bass_text == ""
    assert drums_text == "[Drums: no content]"

    empty = Arrangement(tempo=120.0, ts=(4, 4))
    assert render_parts(empty) == ("", "", "[Drums: no content]")


def test_render_parts_renders_drums_when_there_are_hits():
    arrangement = _arrangement((4, 4), drums=True)
    guitar_text, bass_text, drums_text = render_parts(arrangement)
    assert guitar_text.strip() and bass_text.strip()
    assert drums_text.startswith("Drums")
    assert "kick:" in drums_text
    assert "x" in drums_text
    # The drum grid must not leak into the string parts.
    assert "kick:" not in guitar_text
    assert "kick:" not in bass_text


def test_render_parts_carries_performance_notes_into_the_right_group():
    guitar = build_guitar_track(_notes([20, 64, 59, 55]), 120.0, (4, 4), name="Bass Guitar")
    bass = build_bass_track(_notes([28, 33, 38, 43]), 120.0, (4, 4), name="gtr 2")
    assert guitar.performance_notes
    assert bass.performance_notes == []
    guitar_text, bass_text, _ = render_parts(
        Arrangement(tempo=120.0, ts=(4, 4), guitar=[guitar], bass=[bass])
    )
    assert "Bass Guitar notes:" in guitar_text
    assert "shifted G#0 up 2 octaves" in guitar_text
    assert "notes:" not in bass_text


# --------------------------------------------------------------------------
# MusicXML
# --------------------------------------------------------------------------


def _expected_units(note_el):
    type_el = note_el.find("type")
    assert type_el is not None, "every <note> must carry a <type>"
    units = TYPE_UNITS[type_el.text]
    for _ in note_el.findall("dot"):
        units = units * 3 / 2
    return units


@pytest.mark.parametrize("ts", [(4, 4), (7, 8), (6, 8)])
@pytest.mark.parametrize("start", [0.0, 0.5])
def test_musicxml_measures_fill_the_bar(tmp_path, ts, start):
    arrangement = _arrangement(ts, start=start)
    path = tmp_path / "score.musicxml"
    write_musicxml(arrangement, str(path))

    root = ET.parse(str(path)).getroot()
    parts = root.findall("part")
    # guitar + bass + drums
    assert len(parts) == 3
    bar_units = _bar_units(ts)

    for part in parts:
        measures = part.findall("measure")
        assert measures
        for measure in measures:
            cursor = 0
            for child in measure:
                if child.tag == "note":
                    duration = int(child.find("duration").text)
                    if child.find("chord") is None:
                        cursor += duration
                    assert duration == _expected_units(child), (
                        "measure %s: <duration>%d disagrees with <type>%s"
                        % (measure.get("number"), duration, child.find("type").text)
                    )
                elif child.tag == "backup":
                    cursor -= int(child.find("duration").text)
            assert cursor == bar_units, (
                "part %s measure %s sums to %d, expected %d"
                % (part.get("id"), measure.get("number"), cursor, bar_units)
            )


@pytest.mark.parametrize("ts", [(4, 4), (7, 8), (6, 8)])
def test_musicxml_never_puts_chord_on_a_rest(tmp_path, ts):
    path = tmp_path / "score.musicxml"
    # A half-second lead-in guarantees the first measure opens with rests.
    write_musicxml(_arrangement(ts, start=0.5), str(path))
    root = ET.parse(str(path)).getroot()

    rests = 0
    for note in root.iter("note"):
        if note.find("rest") is None:
            continue
        rests += 1
        assert note.find("chord") is None
    assert rests > 0


def test_musicxml_tempo_direction_in_measure_one(tmp_path):
    path = tmp_path / "score.musicxml"
    write_musicxml(_arrangement((4, 4)), str(path))
    root = ET.parse(str(path)).getroot()

    for part in root.findall("part"):
        measures = part.findall("measure")
        # Guard against a single-bar fixture making "and nowhere else" vacuous.
        assert len(measures) >= 2
        first = measures[0]
        assert first.get("number") == "1"
        metronome = first.find("direction/direction-type/metronome")
        assert metronome is not None
        assert metronome.find("per-minute").text == "120"
        # ...and nowhere else.
        for measure in measures[1:]:
            assert measure.find("direction/direction-type/metronome") is None


def test_musicxml_attributes_only_in_measure_one(tmp_path):
    path = tmp_path / "score.musicxml"
    write_musicxml(_arrangement((4, 4)), str(path))
    root = ET.parse(str(path)).getroot()

    for part in root.findall("part"):
        measures = part.findall("measure")
        # <attributes> used to be re-emitted every bar; with one measure the
        # assertion below would prove nothing.
        assert len(measures) >= 2
        assert measures[0].find("attributes") is not None
        for measure in measures[1:]:
            assert measure.find("attributes") is None


def test_musicxml_percussion_part_uses_unpitched_notes(tmp_path):
    path = tmp_path / "score.musicxml"
    write_musicxml(_arrangement((4, 4)), str(path))
    root = ET.parse(str(path)).getroot()

    drum_part = root.findall("part")[-1]
    unpitched = list(drum_part.iter("unpitched"))
    assert unpitched
    # kick -> F4, snare -> C5, hat -> G5 on the percussion staff.
    displayed = {
        (u.find("display-step").text, u.find("display-octave").text) for u in unpitched
    }
    assert ("F", "4") in displayed
    assert ("C", "5") in displayed
    assert ("G", "5") in displayed
    assert drum_part.find("measure/attributes/clef/sign").text == "percussion"


def test_musicxml_drum_part_presence_follows_the_arrangement(tmp_path):
    # `include_drums` is gone: the drum part exists exactly when there are hits.
    assert "include_drums" not in inspect.signature(write_musicxml).parameters

    with_drums = _arrangement((4, 4), drums=True)
    assert with_drums.drums
    path = tmp_path / "with.musicxml"
    write_musicxml(with_drums, str(path))
    root = ET.parse(str(path)).getroot()
    assert len(root.findall("part")) == 3
    assert list(root.iter("unpitched"))

    without = _arrangement((4, 4), drums=False)
    assert without.drums == []
    path = tmp_path / "without.musicxml"
    write_musicxml(without, str(path))
    root = ET.parse(str(path)).getroot()
    assert len(root.findall("part")) == 2
    assert not list(root.iter("unpitched"))
    assert [sp.get("id") for sp in root.iter("score-part")] == ["P1", "P2"]


@pytest.mark.parametrize("ts", [(4, 4), (7, 8), (6, 8)])
def test_musicxml_backup_separates_the_two_staves(tmp_path, ts):
    path = tmp_path / "score.musicxml"
    write_musicxml(_arrangement(ts), str(path))
    root = ET.parse(str(path)).getroot()
    bar_units = _bar_units(ts)

    # Every assertion below is inside two loops. Without these guards a writer
    # regression that emitted an empty score would turn this test green.
    parts = root.findall("part")
    assert len(parts) == 3  # guitar, bass, drums
    tab_parts = parts[:2]
    assert len(tab_parts) == 2
    for part in tab_parts:
        measures = part.findall("measure")
        assert len(measures) >= 2
        for measure in measures:
            children = list(measure)
            backups = [i for i, c in enumerate(children) if c.tag == "backup"]
            assert len(backups) == 1, "expected exactly one <backup> per measure"
            cut = backups[0]
            assert int(children[cut].find("duration").text) == bar_units
            before = [c for c in children[:cut] if c.tag == "note"]
            after = [c for c in children[cut + 1:] if c.tag == "note"]
            assert before and after
            staves_before = {c.findtext("staff") for c in before}
            staves_after = {c.findtext("staff") for c in after}
            # Rests carry no staff on the drum part, but tab rests do.
            assert staves_before == {"1"}
            assert staves_after == {"2"}
            # Fret/string technicals live only on staff 2.
            assert not any(c.find("notations/technical") is not None for c in before)
            assert any(c.find("notations/technical") is not None for c in after)


def test_musicxml_seven_eight_uses_dotted_forms_not_a_quarter_fallback(tmp_path):
    # A 7/8 bar is 14 units. The old table had no dotted entries and fell back to
    # "quarter" for anything that was not a power of two, so a 14-unit rest was
    # written as a quarter. Prove the dotted forms are actually exercised, which
    # keeps test_musicxml_measures_fill_the_bar's type check non-vacuous.
    path = tmp_path / "score.musicxml"
    # A one-bar lead-in of silence guarantees a full 14-unit rest somewhere.
    write_musicxml(_arrangement((7, 8), start=1.75), str(path))
    root = ET.parse(str(path)).getroot()

    dotted = [n for n in root.iter("note") if n.find("dot") is not None]
    assert dotted, "no dotted durations emitted for 7/8"
    for note in dotted:
        assert int(note.find("duration").text) == _expected_units(note)
    # And a dotted-half (12 units) rest specifically -- the head of a 14-unit gap.
    assert any(
        n.findtext("type") == "half"
        and n.find("dot") is not None
        and n.findtext("duration") == "12"
        for n in root.iter("note")
    )


def test_musicxml_every_note_carries_a_type(tmp_path):
    path = tmp_path / "score.musicxml"
    write_musicxml(_arrangement((7, 8)), str(path))
    root = ET.parse(str(path)).getroot()
    notes = list(root.iter("note"))
    assert notes
    for note in notes:
        assert note.findtext("type") in TYPE_UNITS
        assert int(note.findtext("duration")) == _expected_units(note)


# --------------------------------------------------------------------------
# drum time quantisation -- drum_events and drum_bars must agree
# --------------------------------------------------------------------------


def _late_hit_arrangement():
    """Drum hits at 0.0, 2.0 and 3e-5 s *before* the 4.0 s bar line.

    120 bpm 4/4 gives 2.0 s bars, so the last hit rounds up onto the start of
    bar 3. Grouping events on a rounded start while sizing the bar list from
    the raw start put that hit past the final bar, where it was silently
    dropped from the GP5 and MusicXML exports -- but not from ASCII, which
    laid its grid out differently. Two exports of one arrangement disagreed.
    """
    return build_arrangement(
        [
            Instrument(
                name="guitar",
                notes=_notes([40, 45, 50, 55, 59, 64] * 4, step=0.25),
            ),
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


def test_drum_events_and_drum_bars_quantise_on_the_same_grid():
    assert DRUM_TIME_PLACES == 4

    arrangement = _late_hit_arrangement()
    hits = arrangement.drums
    assert len(hits) == 3

    events = drum_events(hits, lambda h: h.part)
    bars = drum_bars(hits, arrangement.tempo, arrangement.ts)

    # The near-miss start is rounded onto the bar line...
    assert [e.start for e in events] == [0.0, 2.0, 4.0]
    # ...so the bar list has to be sized from the rounded time too, or the
    # last event has no bar to live in.
    assert len(bars) == 3
    assert bars[-1] == (4.0, 6.0)
    assert events[-1].start < bars[-1][1]


def test_ascii_keeps_a_drum_hit_that_rounds_onto_a_bar_line():
    # ASCII was the renderer that never lost this hit -- it is the reference the
    # other two now match on survival. It still *places* the hit from the raw
    # start (last cell of bar 2) rather than the rounded one, so the cell index
    # is deliberately not asserted here; only that all three hits are drawn.
    _, _, drums_text = render_parts(_late_hit_arrangement())
    kick_rows = [ln for ln in drums_text.splitlines() if ln.strip().startswith("kick:")]
    assert len(kick_rows) == 1
    assert kick_rows[0].count("x") == 3
    # Three bars of grid, matching drum_bars.
    assert kick_rows[0].split(": ", 1)[1].count("|") == 2


def test_musicxml_keeps_a_drum_hit_that_rounds_onto_a_bar_line(tmp_path):
    path = tmp_path / "score.musicxml"
    write_musicxml(_late_hit_arrangement(), str(path))
    root = ET.parse(str(path)).getroot()

    drum_part = root.findall("part")[-1]
    assert drum_part.find("measure/attributes/clef/sign").text == "percussion"
    assert len(drum_part.findall("measure")) == 3
    unpitched = list(drum_part.iter("unpitched"))
    assert len(unpitched) == 3, "the hit on the bar line was dropped"
    # One per measure: the last one is the first beat of measure 3.
    per_measure = [
        len(list(measure.iter("unpitched"))) for measure in drum_part.findall("measure")
    ]
    assert per_measure == [1, 1, 1]
