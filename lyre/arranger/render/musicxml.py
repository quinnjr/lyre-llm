from xml.etree import ElementTree as ET

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DURATION_TYPE = {16: "whole", 8: "half", 4: "quarter", 2: "eighth", 1: "sixteenth"}


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


def _note_el(pitch, duration, staff, is_chord, technical=None):
    note = ET.Element("note")
    if is_chord:
        note.append(_el("chord"))
    step, alter, octave = _pitch(pitch)
    p = ET.SubElement(note, "pitch")
    p.append(_el("step", step))
    if alter:
        p.append(_el("alter", alter))
    p.append(_el("octave", octave))
    note.append(_el("duration", duration))
    note.append(_el("voice", 1))
    note.append(_el("type", DURATION_TYPE.get(duration, "quarter")))
    note.append(_el("staff", staff))
    if technical:
        notations = ET.SubElement(note, "notations")
        t = ET.SubElement(notations, "technical")
        t.append(_el("string", technical[0]))
        t.append(_el("fret", technical[1]))
    return note


def _rest_el(duration, staff, is_chord):
    note = ET.Element("note")
    if is_chord:
        note.append(_el("chord"))
    note.append(_el("rest"))
    note.append(_el("duration", duration))
    note.append(_el("voice", 1))
    note.append(_el("type", DURATION_TYPE.get(duration, "quarter")))
    note.append(_el("staff", staff))
    return note


def _attributes(track, ts):
    attrs = ET.Element("attributes")
    attrs.append(_el("divisions", 4))
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


def _measure_el(track, number, tempo, ts):
    measure = ET.Element("measure", {"number": str(number)})
    measure.append(_attributes(track, ts))
    if not track.bars:
        return measure
    b_start, b_end = track.bars[min(number - 1, len(track.bars) - 1)]
    bar_events = [e for e in track.events if b_start <= e.start < b_end]
    bar_events.sort(key=lambda e: e.start)
    beat_sec = 60.0 / tempo
    duration_beats = (b_end - b_start) / beat_sec
    total_units = int(round(duration_beats * 4))
    current_units = 0
    for event in bar_events:
        start_units = int(round((event.start - b_start) / beat_sec * 4))
        if start_units > current_units:
            gap = start_units - current_units
            measure.append(_rest_el(gap, 1, False))
            measure.append(_rest_el(gap, 2, True))
            current_units = start_units
        next_start = min((e.start for e in bar_events if e.start > event.start + 1e-6), default=event.start + (b_end - event.start))
        dur_units = int(round((next_start - event.start) / beat_sec * 4))
        dur_units = max(1, min(dur_units, total_units - current_units))
        for i, n in enumerate(sorted(event.notes, key=lambda x: x.pitch, reverse=True)):
            technical = (n.string, n.fret)
            measure.append(_note_el(n.pitch, dur_units, 1, i > 0))
            measure.append(_note_el(n.pitch, dur_units, 2, True, technical=technical))
        current_units += dur_units
    if current_units < total_units:
        rem = total_units - current_units
        measure.append(_rest_el(rem, 1, False))
        measure.append(_rest_el(rem, 2, True))
    return measure


def write_musicxml(arrangement, path):
    root = ET.Element("score-partwise", {"version": "3.1"})
    part_list = ET.SubElement(root, "part-list")
    tracks = list(arrangement.guitar) + list(arrangement.bass)
    for i, track in enumerate(tracks, start=1):
        sp = ET.SubElement(part_list, "score-part", {"id": f"P{i}"})
        sp.append(_el("part-name", track.name))
        si = ET.SubElement(sp, "score-instrument", {"id": f"P{i}-I1"})
        si.append(_el("instrument-name", track.name))
    for i, track in enumerate(tracks, start=1):
        part = ET.SubElement(root, "part", {"id": f"P{i}"})
        n_bars = len(track.bars) or 1
        for bar_idx in range(1, n_bars + 1):
            part.append(_measure_el(track, bar_idx, arrangement.tempo, arrangement.ts))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
