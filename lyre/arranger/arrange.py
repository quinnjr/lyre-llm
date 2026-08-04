# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import re
from dataclasses import dataclass, field

from lyre.errors import LyreError

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

GUITAR_TUNING = [40, 45, 50, 55, 59, 64]
BASS_TUNING = [28, 33, 38, 43]

GUITAR_MAX_FRET = 24
BASS_MAX_FRET = 21  # a 4-string bass does not have 24 frets
DEFAULT_MAX_SPAN = 4  # frets a hand can reach without shifting position

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


_NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)?$")


def note_to_midi(name):
    """Parse a scientific-pitch note name (``C4``, ``Db3``, ``eb2``) to MIDI."""
    match = _NOTE_RE.match(str(name).strip())
    if match is None:
        raise ValueError(f"unknown note name {name!r}")
    letter, accidental, octave = match.groups()
    pitch_class = _NATURAL_PC[letter.upper()]
    if accidental == "#":
        pitch_class += 1
    elif accidental == "b":
        pitch_class -= 1
    octave = 4 if octave is None else int(octave)
    return pitch_class + (octave + 1) * 12


def string_label(pitch):
    return NOTE_NAMES[pitch % 12]


def pitch_name(pitch):
    """Human-readable scientific pitch name, e.g. 36 -> ``C2``."""
    pitch = int(pitch)
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def resolve_tuning(spec, default, presets=None):
    presets = presets or {}
    if spec is None:
        return list(default)
    if isinstance(spec, (list, tuple)):
        # A bad note name in the list is the same class of user error as an
        # unknown preset below, so it gets the same exception type -- otherwise
        # `[E2, A2, Q9]` is a traceback while `drop-q` is a clean message.
        out = []
        for s in spec:
            if isinstance(s, int):
                out.append(int(s))
                continue
            try:
                out.append(note_to_midi(str(s)))
            except ValueError as exc:
                raise LyreError(f"unknown tuning {spec!r}: {exc}") from exc
        return out
    key = str(spec).strip().lower().replace(" ", "-")
    if key in presets:
        return list(presets[key])
    raise LyreError(f"unknown tuning {spec!r}; use a preset or a list of pitches")


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
    """A tab track.

    ``performance_notes`` holds human-readable notes describing every
    reduction the arranger had to make (octave shifts, dropped voices).
    Renderers are expected to surface these to the user.
    """

    name: str
    tuning: list
    max_fret: int
    events: list = field(default_factory=list)
    bars: list = field(default_factory=list)
    performance_notes: list = field(default_factory=list)


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


def _search_voicing(pitches, tuning, prev_frets, max_span, max_fret):
    """Best fretting for exactly ``pitches`` (sorted high to low), or None.

    Score is ``(span, melody_penalty, movement)``: fret span dominates, then how
    far the lead voice sits from the highest string, then voice-leading motion.
    """
    top_index = len(tuning) - 1
    best = None

    def rec(i, used, frets, span_lo, span_hi, top_string):
        nonlocal best
        if best is not None and span_hi - span_lo > best[0]:
            return
        if i == len(pitches):
            if span_hi - span_lo > max_span:
                return
            movement = sum(
                abs(f - prev_frets[s]) for s, f in enumerate(frets) if f >= 0
            )
            melody = top_index - top_string if top_string >= 0 else top_index
            score = (span_hi - span_lo, melody, movement)
            if best is None or score < best[:3]:
                best = (span_hi - span_lo, melody, movement, frets[:])
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
            rec(
                i + 1,
                used,
                frets,
                min(span_lo, fret),
                max(span_hi, fret),
                s if i == 0 else top_string,
            )
            frets[s] = -1
            used.remove(s)

    rec(0, set(), [-1] * len(tuning), 1000, -1, -1)
    if best is None:
        return None
    return best[3]


def _fit_octave(pitch, lo, hi):
    """Shift ``pitch`` by whole octaves into ``[lo, hi]``.

    Returns ``(pitch, octaves_shifted)`` or ``(None, 0)`` when impossible.
    """
    shifted = int(pitch)
    octaves = 0
    while shifted < lo:
        shifted += 12
        octaves += 1
    while shifted > hi:
        shifted -= 12
        octaves -= 1
    if shifted < lo or shifted > hi:
        return None, 0
    return shifted, octaves


def voice_chord(
    pitches,
    tuning,
    prev_frets=None,
    max_span=DEFAULT_MAX_SPAN,
    max_fret=GUITAR_MAX_FRET,
    report=None,
):
    """Fret a simultaneity, reducing it until it is playable.

    Pitches outside the instrument range are octave-shifted; if there are still
    more pitches than strings (or the reach is impossible) the lowest voices are
    dropped so the lead melody survives. Every such compromise is appended to
    ``report`` as a human-readable string when a list is supplied.
    """
    prev_frets = prev_frets or [0] * len(tuning)
    wanted = sorted({int(p) for p in pitches}, reverse=True)
    if not wanted:
        return None

    lo = min(tuning)
    hi = max(tuning) + max_fret
    shifts = {}
    placed = []
    n_unfittable = 0
    for pitch in wanted:
        fitted, octaves = _fit_octave(pitch, lo, hi)
        if fitted is None:
            n_unfittable += 1
            continue
        if octaves and fitted not in shifts:
            shifts[fitted] = "shifted %s %s %d octave%s" % (
                pitch_name(pitch),
                "up" if octaves > 0 else "down",
                abs(octaves),
                "" if abs(octaves) == 1 else "s",
            )
        placed.append(fitted)

    placed = sorted(set(placed), reverse=True)
    # Count against the folded, de-duplicated set: an octave-doubled voice that
    # collapses onto an existing pitch is not a musical loss, so reporting it as
    # a "dropped note" would be a false alarm on every power chord.
    n_requested = len(placed) + n_unfittable
    if len(placed) > len(tuning):
        placed = placed[: len(tuning)]

    frets = None
    while placed:
        frets = _search_voicing(placed, tuning, prev_frets, max_span, max_fret)
        if frets is not None:
            break
        placed = placed[:-1]
    if frets is None:
        return None

    if report is not None:
        report.extend(shifts[p] for p in placed if p in shifts)
        if len(placed) < n_requested:
            report.append(
                "dropped %d of %d simultaneous notes"
                % (n_requested - len(placed), n_requested)
            )
    return frets


def events_from_notes(notes, tolerance=0.02, sort=True):
    """Group notes into simultaneities: onsets within ``tolerance`` seconds."""
    if sort:
        notes = sorted(notes, key=lambda n: n.start)
    events = []
    for note in notes:
        if events and abs(note.start - events[-1][0]) <= tolerance:
            events[-1][1].append(note)
        else:
            events.append((note.start, [note]))
    return events


def _source_end(group, pitch, b_end):
    """End time of the source note(s) that produced ``pitch``, clipped to the bar."""
    ends = [n.end for n in group if n.pitch == pitch]
    if not ends:
        ends = [n.end for n in group if (n.pitch - pitch) % 12 == 0]
    if not ends:
        return b_end
    return min(max(ends), b_end)


def _build_tab_track(notes, tempo, ts, tuning, name, max_span, max_fret):
    track = TabTrack(name=name, tuning=tuning, max_fret=max_fret)
    if not notes:
        return track
    notes = sorted(notes, key=lambda n: n.start)
    n_notes = len(notes)
    duration = max(n.end for n in notes)
    bars = _bar_boundaries(duration, tempo, ts)
    prev_frets = [0] * len(tuning)
    index = 0
    for bar_no, (b_start, b_end) in enumerate(bars, start=1):
        last_bar = bar_no == len(bars)
        stop = index
        while stop < n_notes and (
            notes[stop].start < b_end or (last_bar and notes[stop].start <= b_end)
        ):
            stop += 1
        bar_notes = notes[index:stop]
        index = stop
        for start, group in events_from_notes(bar_notes, sort=False):
            pitches = [n.pitch for n in group]
            report = []
            frets = voice_chord(
                pitches, tuning, prev_frets, max_span, max_fret, report=report
            )
            if frets is None:
                track.performance_notes.append(
                    "bar %d: dropped %d unplayable note%s"
                    % (bar_no, len(pitches), "" if len(pitches) == 1 else "s")
                )
                continue
            for message in report:
                track.performance_notes.append("bar %d: %s" % (bar_no, message))
            prev_frets = [f if f >= 0 else p for f, p in zip(frets, prev_frets)]
            event = TabEvent(start=start)
            for s, f in enumerate(frets):
                if f < 0:
                    continue
                pitch = tuning[s] + f
                event.notes.append(
                    TabNote(
                        string=s,
                        fret=f,
                        start=start,
                        end=_source_end(group, pitch, b_end),
                        pitch=pitch,
                    )
                )
            track.events.append(event)
        track.bars.append((b_start, b_end))
    return track


def build_guitar_track(
    notes,
    tempo,
    ts=(4, 4),
    tuning=None,
    name="Guitar",
    max_span=DEFAULT_MAX_SPAN,
    max_fret=GUITAR_MAX_FRET,
):
    return _build_tab_track(
        notes,
        tempo,
        ts,
        resolve_tuning(tuning, GUITAR_TUNING, GUITAR_TUNINGS),
        name,
        max_span,
        max_fret,
    )


def build_bass_track(
    notes,
    tempo,
    ts=(4, 4),
    tuning=None,
    name="Bass",
    max_span=DEFAULT_MAX_SPAN,
    max_fret=BASS_MAX_FRET,
):
    return _build_tab_track(
        notes,
        tempo,
        ts,
        resolve_tuning(tuning, BASS_TUNING, BASS_TUNINGS),
        name,
        max_span,
        max_fret,
    )


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
    arrangement.drums.sort(key=lambda h: h.start)
    if arrange_melodic_to_guitar and not arrangement.guitar and melodic:
        melodic.sort(key=lambda i: len(i.notes), reverse=True)
        merged = [n for instrument in melodic for n in instrument.notes]
        name = melodic[0].name
        track = build_guitar_track(
            merged, tempo, arrangement.ts, tuning=guitar_tuning, name=f"{name} (arranged)"
        )
        if len(melodic) > 1:
            track.performance_notes.insert(
                0,
                "arranged from %d melodic parts: %s"
                % (len(melodic), ", ".join(i.name for i in melodic)),
            )
        arrangement.guitar.append(track)
    return arrangement
