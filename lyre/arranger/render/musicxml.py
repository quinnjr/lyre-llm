from xml.etree import ElementTree as ET

from ._common import bucket_by_bar, drum_bars, drum_events

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Divisions per quarter note. A whole note is therefore 16 units.
DIVISIONS = 4
UNITS_PER_WHOLE = 4 * DIVISIONS

# (units, MusicXML <type>, number of dots), largest first, so a greedy
# decomposition of any positive duration terminates on the 1-unit entry.
DURATION_FORMS = [
    (24, "whole", 1),
    (16, "whole", 0),
    (12, "half", 1),
    (8, "half", 0),
    (6, "quarter", 1),
    (4, "quarter", 0),
    (3, "eighth", 1),
    (2, "eighth", 0),
    (1, "16th", 0),
]

# Where each classified drum part sits on a 5-line percussion staff, and the
# General MIDI key it sounds.
DRUM_DISPLAY = {
    "kick": ("F", 4, 36),
    "snare": ("C", 5, 38),
    "hat": ("G", 5, 42),
    "crash": ("A", 5, 49),
    "ride": ("F", 5, 51),
    "tom": ("E", 5, 45),
}
DRUM_PARTS = ["kick", "snare", "hat", "crash", "ride", "tom"]

# _decompose's greedy loop only terminates because the smallest form is one
# unit, which is also why its `else` branch is marked as needing no coverage.
# Checked with a real raise, not an assert: `python -O` strips asserts, and
# with it the guarantee that branch relies on.
if DURATION_FORMS[-1][0] != 1:
    raise ValueError("DURATION_FORMS must end at a 1-unit form for _decompose to terminate")


def _pitch(midi):
    name = NOTE_NAMES[midi % 12]
    step = name[0]
    alter = 1 if "#" in name else 0
    octave = midi // 12 - 1
    return step, alter, octave


def _el(tag, text=None, attrs=None):
    el = ET.Element(tag, attrs or {})
    if text is not None:
        el.text = str(text)
    return el


def _decompose(units):
    """Split a duration into tied power-of-two (optionally dotted) components."""
    units = int(units)
    parts = []
    while units > 0:
        for value, name, dots in DURATION_FORMS:
            if value <= units:
                parts.append((value, name, dots))
                units -= value
                break
        else:  # pragma: no cover - DURATION_FORMS ends at 1 so this cannot hit
            break
    return parts


def _append_type(note, name, dots):
    note.append(_el("type", name))
    for _ in range(dots):
        note.append(_el("dot"))


def _append_ties(note, tie_start, tie_stop):
    # <tie> is the sound half, <tied> inside <notations> is the notation half.
    if tie_stop:
        note.append(_el("tie", attrs={"type": "stop"}))
    if tie_start:
        note.append(_el("tie", attrs={"type": "start"}))


def _notations(note, tie_start, tie_stop, technical=None):
    if not (tie_start or tie_stop or technical):
        return
    notations = ET.SubElement(note, "notations")
    if tie_stop:
        ET.SubElement(notations, "tied", {"type": "stop"})
    if tie_start:
        ET.SubElement(notations, "tied", {"type": "start"})
    if technical:
        t = ET.SubElement(notations, "technical")
        t.append(_el("string", technical[0]))
        t.append(_el("fret", technical[1]))


def _note_el(pitch, units, name, dots, staff, is_chord,
             technical=None, tie_start=False, tie_stop=False):
    note = ET.Element("note")
    if is_chord:
        note.append(_el("chord"))
    step, alter, octave = _pitch(pitch)
    p = ET.SubElement(note, "pitch")
    p.append(_el("step", step))
    if alter:
        p.append(_el("alter", alter))
    p.append(_el("octave", octave))
    note.append(_el("duration", units))
    _append_ties(note, tie_start, tie_stop)
    note.append(_el("voice", 1))
    _append_type(note, name, dots)
    note.append(_el("staff", staff))
    _notations(note, tie_start, tie_stop, technical)
    return note


def _unpitched_note_el(part, units, name, dots, tie_start=False, tie_stop=False):
    step, octave, _midi = DRUM_DISPLAY[part]
    note = ET.Element("note")
    u = ET.SubElement(note, "unpitched")
    u.append(_el("display-step", step))
    u.append(_el("display-octave", octave))
    note.append(_el("duration", units))
    _append_ties(note, tie_start, tie_stop)
    note.append(_el("instrument", attrs={"id": f"DRUMS-{part}"}))
    note.append(_el("voice", 1))
    _append_type(note, name, dots)
    _notations(note, tie_start, tie_stop)
    return note


def _rest_els(units, staff):
    """A rest of `units`, split into notatable components."""
    out = []
    for value, name, dots in _decompose(units):
        note = ET.Element("note")
        note.append(_el("rest"))
        note.append(_el("duration", value))
        note.append(_el("voice", 1))
        _append_type(note, name, dots)
        if staff is not None:
            note.append(_el("staff", staff))
        out.append(note)
    return out


def _backup_el(units):
    backup = ET.Element("backup")
    backup.append(_el("duration", units))
    return backup


def _attributes(track, ts):
    attrs = ET.Element("attributes")
    attrs.append(_el("divisions", DIVISIONS))
    key = ET.SubElement(attrs, "key")
    key.append(_el("fifths", 0))
    time = ET.SubElement(attrs, "time")
    time.append(_el("beats", ts[0]))
    time.append(_el("beat-type", ts[1]))
    attrs.append(_el("staves", 2))
    c1 = ET.SubElement(attrs, "clef", {"number": "1"})
    c1.append(_el("sign", "G"))
    c1.append(_el("line", 2))
    c2 = ET.SubElement(attrs, "clef", {"number": "2"})
    c2.append(_el("sign", "TAB"))
    c2.append(_el("line", 5))
    sd = ET.SubElement(attrs, "staff-details", {"number": "2"})
    sd.append(_el("staff-lines", len(track.tuning)))
    for i, pitch in enumerate(reversed(track.tuning), start=1):
        st = ET.SubElement(sd, "staff-tuning", {"line": str(i)})
        step, alter, octave = _pitch(pitch)
        st.append(_el("tuning-step", step))
        if alter:
            st.append(_el("tuning-alter", alter))
        st.append(_el("tuning-octave", octave))
    return attrs


def _percussion_attributes(ts):
    attrs = ET.Element("attributes")
    attrs.append(_el("divisions", DIVISIONS))
    time = ET.SubElement(attrs, "time")
    time.append(_el("beats", ts[0]))
    time.append(_el("beat-type", ts[1]))
    clef = ET.SubElement(attrs, "clef")
    clef.append(_el("sign", "percussion"))
    clef.append(_el("line", 2))
    return attrs


def _tempo_el(tempo):
    direction = ET.Element("direction", {"placement": "above"})
    dtype = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(dtype, "metronome")
    metronome.append(_el("beat-unit", "quarter"))
    metronome.append(_el("per-minute", int(round(tempo))))
    direction.append(_el("sound", attrs={"tempo": str(round(float(tempo), 4))}))
    return direction


def _bar_units(ts):
    """Nominal length of a full measure, in divisions.

    Every measure is written to the full nominal length (matching the GP5
    renderer) so that importers see bars that fill; the arranger's final bar
    can be short in wall-clock terms and is padded with rests.
    """
    return ts[0] * UNITS_PER_WHOLE // ts[1]


def _measure_el(track, number, tempo, ts, bar_events, bar, emit_attributes):
    measure = ET.Element("measure", {"number": str(number)})
    if emit_attributes:
        measure.append(_attributes(track, ts))
    if number == 1:
        measure.append(_tempo_el(tempo))
    if bar is None:
        return measure
    b_start, b_end = bar
    beat_sec = 60.0 / tempo
    total_units = _bar_units(ts)
    staff1 = []
    staff2 = []
    current_units = 0
    for i, event in enumerate(bar_events):
        start_units = int(round((event.start - b_start) / beat_sec * DIVISIONS))
        if start_units > current_units:
            gap = min(start_units, total_units) - current_units
            if gap > 0:
                staff1.extend(_rest_els(gap, 1))
                staff2.extend(_rest_els(gap, 2))
                current_units += gap
        next_start = bar_events[i + 1].start if i + 1 < len(bar_events) else b_end
        dur_units = int(round((next_start - event.start) / beat_sec * DIVISIONS))
        dur_units = min(dur_units, total_units - current_units)
        if dur_units <= 0:
            continue
        chord = sorted(event.notes, key=lambda x: x.pitch, reverse=True)
        components = _decompose(dur_units)
        for j, (value, name, dots) in enumerate(components):
            tie_start = j < len(components) - 1
            tie_stop = j > 0
            for k, n in enumerate(chord):
                staff1.append(
                    _note_el(n.pitch, value, name, dots, 1, k > 0,
                             tie_start=tie_start, tie_stop=tie_stop)
                )
                staff2.append(
                    _note_el(n.pitch, value, name, dots, 2, k > 0,
                             technical=(n.string, n.fret),
                             tie_start=tie_start, tie_stop=tie_stop)
                )
        current_units += dur_units
    if current_units < total_units:
        rem = total_units - current_units
        staff1.extend(_rest_els(rem, 1))
        staff2.extend(_rest_els(rem, 2))
        current_units = total_units
    for el in staff1:
        measure.append(el)
    if staff2 and current_units > 0:
        # The standard way to write a second staff: rewind the cursor rather
        # than abusing <chord>, which is meaningless on a <rest>.
        measure.append(_backup_el(current_units))
        for el in staff2:
            measure.append(el)
    return measure


def _drum_measure_el(number, tempo, ts, bar_hits, bar, emit_attributes):
    measure = ET.Element("measure", {"number": str(number)})
    if emit_attributes:
        measure.append(_percussion_attributes(ts))
    if number == 1:
        measure.append(_tempo_el(tempo))
    if bar is None:
        return measure
    b_start, b_end = bar
    beat_sec = 60.0 / tempo
    total_units = _bar_units(ts)
    current_units = 0
    for i, event in enumerate(bar_hits):
        start_units = int(round((event.start - b_start) / beat_sec * DIVISIONS))
        if start_units > current_units:
            gap = min(start_units, total_units) - current_units
            if gap > 0:
                for el in _rest_els(gap, None):
                    measure.append(el)
                current_units += gap
        next_start = bar_hits[i + 1].start if i + 1 < len(bar_hits) else b_end
        dur_units = int(round((next_start - event.start) / beat_sec * DIVISIONS))
        dur_units = min(dur_units, total_units - current_units)
        if dur_units <= 0:
            continue
        components = _decompose(dur_units)
        for j, (value, name, dots) in enumerate(components):
            tie_start = j < len(components) - 1
            tie_stop = j > 0
            for k, part in enumerate(event.values):
                note = _unpitched_note_el(part, value, name, dots, tie_start, tie_stop)
                if k > 0:
                    note.insert(0, _el("chord"))
                measure.append(note)
        current_units += dur_units
    if current_units < total_units:
        for el in _rest_els(total_units - current_units, None):
            measure.append(el)
    return measure


def _drum_part(hit):
    """Payload extractor for _common.drum_events: the part name, if notatable."""
    return hit.part if hit.part in DRUM_DISPLAY else None


def write_musicxml(arrangement, path):
    root = ET.Element("score-partwise", {"version": "3.1"})
    part_list = ET.SubElement(root, "part-list")
    tracks = list(arrangement.guitar) + list(arrangement.bass)
    for i, track in enumerate(tracks, start=1):
        sp = ET.SubElement(part_list, "score-part", {"id": f"P{i}"})
        sp.append(_el("part-name", track.name))
        si = ET.SubElement(sp, "score-instrument", {"id": f"P{i}-I1"})
        si.append(_el("instrument-name", track.name))
    drum_hits = list(arrangement.drums)
    drum_id = f"P{len(tracks) + 1}"
    if drum_hits:
        sp = ET.SubElement(part_list, "score-part", {"id": drum_id})
        sp.append(_el("part-name", "Drums"))
        for part in DRUM_PARTS:
            step, octave, midi = DRUM_DISPLAY[part]
            si = ET.SubElement(sp, "score-instrument", {"id": f"DRUMS-{part}"})
            si.append(_el("instrument-name", part))
            mi = ET.SubElement(sp, "midi-instrument", {"id": f"DRUMS-{part}"})
            mi.append(_el("midi-channel", 10))
            mi.append(_el("midi-unpitched", midi + 1))
    for i, track in enumerate(tracks, start=1):
        part = ET.SubElement(root, "part", {"id": f"P{i}"})
        buckets = bucket_by_bar(track.events, track.bars)
        n_bars = len(track.bars) or 1
        for bar_idx in range(n_bars):
            bar = track.bars[bar_idx] if bar_idx < len(track.bars) else None
            events = buckets[bar_idx] if bar_idx < len(buckets) else []
            # <attributes> only in measure 1: time signature and tuning are
            # fixed for the whole arrangement, so they never change mid-score.
            part.append(
                _measure_el(
                    track, bar_idx + 1, arrangement.tempo, arrangement.ts,
                    events, bar, bar_idx == 0,
                )
            )
    if drum_hits:
        part = ET.SubElement(root, "part", {"id": drum_id})
        bars = drum_bars(drum_hits, arrangement.tempo, arrangement.ts)
        events = drum_events(drum_hits, _drum_part, sort_key=DRUM_PARTS.index)
        buckets = bucket_by_bar(events, bars)
        n_bars = len(bars) or 1
        for bar_idx in range(n_bars):
            bar = bars[bar_idx] if bar_idx < len(bars) else None
            hits = buckets[bar_idx] if bar_idx < len(buckets) else []
            part.append(
                _drum_measure_el(
                    bar_idx + 1, arrangement.tempo, arrangement.ts,
                    hits, bar, bar_idx == 0,
                )
            )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
