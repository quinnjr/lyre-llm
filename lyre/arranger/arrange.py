from dataclasses import dataclass, field

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]
BASS_TUNING = [28, 33, 38, 43]

GUITAR_TUNINGS = {
    "standard": GUITAR_TUNING,
    "half-step-down": [39, 44, 49, 54, 58, 63],
    "whole-step-down": [38, 43, 48, 53, 57, 62],
    "drop-d": [38, 45, 50, 55, 59, 64],
    "drop-c": [36, 45, 50, 55, 59, 64],
    "drop-b": [35, 45, 50, 55, 59, 64],
    "drop-a": [33, 45, 50, 55, 59, 64],
}

BASS_TUNINGS = {
    "standard": BASS_TUNING,
    "half-step-down": [27, 32, 37, 42],
    "whole-step-down": [26, 31, 36, 41],
    "drop-d": [26, 33, 38, 43],
    "drop-c": [24, 33, 38, 43],
    "drop-b": [23, 33, 38, 43],
}


def note_to_midi(name):
    name = name.strip().replace("b", "#")
    if name.endswith("#"):
        name = name + "4"
    octave = 4
    if name and name[-1].isdigit():
        octave = int(name[-1])
        name = name[:-1]
    if name not in NOTE_NAMES:
        raise ValueError(f"unknown note name {name!r}")
    return NOTE_NAMES.index(name) + (octave + 1) * 12


def string_label(pitch):
    return NOTE_NAMES[pitch % 12]


def resolve_tuning(spec, default, presets=None):
    presets = presets or {}
    if spec is None:
        return list(default)
    if isinstance(spec, (list, tuple)):
        return [int(s) if isinstance(s, int) else note_to_midi(str(s)) for s in spec]
    key = str(spec).strip().lower().replace(" ", "-")
    if key in presets:
        return list(presets[key])
    raise ValueError(f"unknown tuning {spec!r}; use a preset or a list of pitches")


@dataclass
class TabNote:
    string: int
    fret: int
    start: float
    end: float
    pitch: int


@dataclass
class TabEvent:
    start: float
    notes: list = field(default_factory=list)


@dataclass
class TabTrack:
    name: str
    tuning: list
    max_fret: int
    events: list = field(default_factory=list)
    bars: list = field(default_factory=list)


@dataclass
class DrumHit:
    start: float
    part: str


@dataclass
class Arrangement:
    tempo: float
    ts: tuple = (4, 4)
    guitar: list = field(default_factory=list)
    bass: list = field(default_factory=list)
    drums: list = field(default_factory=list)


def bar_seconds(tempo, ts):
    numerator, denominator = ts
    return numerator * (4.0 / denominator) * (60.0 / tempo)


def _bar_boundaries(duration, tempo, ts):
    bar_sec = bar_seconds(tempo, ts)
    bars = []
    t = 0.0
    while t < duration - 1e-6:
        end = min(t + bar_sec, duration)
        bars.append((t, end))
        t = end
    return bars


def voice_chord(
    pitches,
    tuning,
    prev_frets=None,
    max_span=4,
    max_fret=24,
):
    prev_frets = prev_frets or [0] * len(tuning)
    pitches = sorted(set(pitches), reverse=True)
    best = None

    def rec(i, used, frets, span_lo, span_hi):
        nonlocal best
        if best is not None and span_hi - span_lo > best[0]:
            return
        if i == len(pitches):
            if span_hi - span_lo > max_span:
                return
            movement = sum(
                abs(f - prev_frets[s]) for s, f in enumerate(frets) if f >= 0
            )
            score = (span_hi - span_lo, movement)
            if best is None or score < best[:2]:
                best = (span_hi - span_lo, movement, frets[:])
            return
        pitch = pitches[i]
        for s in range(len(tuning)):
            if s in used:
                continue
            fret = pitch - tuning[s]
            if fret < 0 or fret > max_fret:
                continue
            used.add(s)
            frets[s] = fret
            rec(i + 1, used, frets, min(span_lo, fret), max(span_hi, fret))
            frets[s] = -1
            used.remove(s)

    frets = [-1] * len(tuning)
    rec(0, set(), frets, 1000, -1)
    if best is None:
        return None
    return best[2]


def _events_from_notes(notes, tolerance=0.02):
    notes = sorted(notes, key=lambda n: n.start)
    events = []
    for note in notes:
        if events and abs(note.start - events[-1][0]) <= tolerance:
            events[-1][1].append(note)
        else:
            events.append((note.start, [note]))
    return events


def build_guitar_track(notes, tempo, ts=(4, 4), tuning=None, name="Guitar", max_span=4, max_fret=24):
    tuning = resolve_tuning(tuning, GUITAR_TUNING, GUITAR_TUNINGS)
    track = TabTrack(name=name, tuning=tuning, max_fret=max_fret)
    if not notes:
        return track
    duration = max(n.end for n in notes)
    bars = _bar_boundaries(duration, tempo, ts)
    prev_frets = [0] * len(tuning)
    for b_start, b_end in bars:
        bar_notes = [n for n in notes if n.end > b_start and n.start < b_end]
        for start, group in _events_from_notes(bar_notes):
            pitches = [n.pitch for n in group]
            frets = voice_chord(pitches, tuning, prev_frets, max_span, max_fret)
            if frets is None:
                continue
            prev_frets = [f if f >= 0 else p for f, p in zip(frets, prev_frets)]
            event = TabEvent(start=start)
            for s, f in enumerate(frets):
                if f >= 0:
                    event.notes.append(TabNote(string=s, fret=f, start=start, end=b_end, pitch=tuning[s] + f))
            track.events.append(event)
        track.bars.append((b_start, b_end))
    return track


def build_bass_track(notes, tempo, ts=(4, 4), tuning=None, name="Bass", max_span=4, max_fret=21):
    tuning = resolve_tuning(tuning, BASS_TUNING, BASS_TUNINGS)
    track = TabTrack(name=name, tuning=tuning, max_fret=max_fret)
    if not notes:
        return track
    duration = max(n.end for n in notes)
    bars = _bar_boundaries(duration, tempo, ts)
    prev_frets = [0] * len(tuning)
    for b_start, b_end in bars:
        bar_notes = [n for n in notes if n.end > b_start and n.start < b_end]
        for start, group in _events_from_notes(bar_notes):
            pitches = [n.pitch for n in group]
            frets = voice_chord(pitches, tuning, prev_frets, max_span, max_fret)
            if frets is None:
                continue
            prev_frets = [f if f >= 0 else p for f, p in zip(frets, prev_frets)]
            event = TabEvent(start=start)
            for s, f in enumerate(frets):
                if f >= 0:
                    event.notes.append(TabNote(string=s, fret=f, start=start, end=b_end, pitch=tuning[s] + f))
            track.events.append(event)
        track.bars.append((b_start, b_end))
    return track


DRUM_MAP = {
    "kick": {36},
    "snare": {38, 40},
    "hat": {42, 44, 46, 54},
    "crash": {49, 55, 57},
    "ride": {51, 59},
    "tom": {41, 43, 45, 47, 48, 50},
}


def classify_drums(notes):
    hits = []
    for n in notes:
        for part, pitches in DRUM_MAP.items():
            if n.pitch in pitches:
                hits.append(DrumHit(start=n.start, part=part))
                break
    hits.sort(key=lambda h: h.start)
    return hits


def build_arrangement(
    instruments,
    tempo=120.0,
    time_signature=(4, 4),
    guitar_tuning=None,
    bass_tuning=None,
    arrange_melodic_to_guitar=True,
):
    arrangement = Arrangement(tempo=tempo, ts=tuple(time_signature))
    melodic = []
    for instrument in instruments:
        family = instrument.family
        if family == "guitar":
            arrangement.guitar.append(
                build_guitar_track(
                    instrument.notes, tempo, arrangement.ts,
                    tuning=guitar_tuning, name=instrument.name,
                )
            )
        elif family == "bass":
            arrangement.bass.append(
                build_bass_track(
                    instrument.notes, tempo, arrangement.ts,
                    tuning=bass_tuning, name=instrument.name,
                )
            )
        elif family == "drums":
            arrangement.drums.extend(classify_drums(instrument.notes))
        else:
            melodic.append(instrument)
    if arrange_melodic_to_guitar and not arrangement.guitar and melodic:
        melodic.sort(key=lambda i: len(i.notes), reverse=True)
        merged = melodic[0].notes
        name = melodic[0].name
        arrangement.guitar.append(
            build_guitar_track(merged, tempo, arrangement.ts, tuning=guitar_tuning, name=f"{name} (arranged)")
        )
    return arrangement
