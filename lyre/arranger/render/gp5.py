from guitarpro import Song, Track, Beat, Note, Duration, TimeSignature
from guitarpro import GuitarString

from ..arrange import GUITAR_TUNING

_DURATIONS = [1, 2, 4, 8, 16, 32, 64]


def _bar_units(numerator, denominator):
    return numerator * 64 // denominator


def _duration_code(units):
    for v in _DURATIONS:
        if 64 // v <= units:
            return v
    return 64


def _split_units(units):
    chosen = []
    guard = 0
    while units > 0:
        v = _duration_code(units)
        step = 64 // v
        if step <= 0:
            v = 64
            step = 1
        chosen.append(v)
        units -= step
        guard += 1
        if guard > 1000:
            break
    return chosen


def _note_to_quarters(note_sec, tempo):
    beat = 60.0 / tempo
    return note_sec / beat


def _quarters_to_units(quarters):
    return int(round(quarters * 4))


def _second_to_units(sec, tempo):
    return _quarters_to_units(_note_to_quarters(sec, tempo))


def write_gp5(arrangement, path):
    numerator, denominator = arrangement.ts
    song = Song(title="lyre transcription", tempo=int(round(arrangement.tempo)))
    time_sig = TimeSignature(
        numerator=numerator,
        denominator=Duration(denominator),
    )
    tracks_meta = [
        (t.name, t.tuning, t.max_fret)
        for t in list(arrangement.guitar) + list(arrangement.bass)
    ]
    if not tracks_meta:
        tracks_meta = [("empty", GUITAR_TUNING, 24)]
    # Song() ships with a default track and header, and Track(song=...) seeds one
    # Measure per existing header. Clear the headers first so every track starts
    # empty and measure counts stay in step with measureHeaders.
    song.measureHeaders = []
    built = []
    for idx, (name, tuning, max_fret) in enumerate(tracks_meta):
        track = Track(song=song, number=idx + 1)
        track.name = name
        track.fretCount = max_fret
        # Guitar Pro numbers strings 1..n from the highest-pitched string down,
        # while our tunings are stored low-to-high.
        track.strings = [
            GuitarString(number=i + 1, value=p)
            for i, p in enumerate(reversed(tuning))
        ]
        built.append(track)
    song.tracks = built
    all_tracks = list(arrangement.guitar) + list(arrangement.bass)
    n_bars = max((len(t.bars) for t in all_tracks), default=1) or 1
    bar_units = _bar_units(numerator, denominator)
    for i in range(n_bars):
        song.newMeasure()
        header = song.measureHeaders[i]
        header.timeSignature = time_sig
        header.number = i + 1
    for g_idx, tab_track in enumerate(all_tracks):
        for bar_idx, (b_start, b_end) in enumerate(tab_track.bars):
            measure = song.tracks[g_idx].measures[bar_idx]
            # GP5 always stores Measure.maxVoices voices; replacing the list with a
            # single voice desynchronises the reader (it fails on "voice 2").
            voice = measure.voices[0]
            voice.beats = []
            events = [e for e in tab_track.events if b_start <= e.start < b_end]
            events.sort(key=lambda e: e.start)
            cursor = 0
            for i, event in enumerate(events):
                ev_start = _second_to_units(event.start - b_start, arrangement.tempo)
                gap = ev_start - cursor
                for v in _split_units(gap):
                    voice.beats.append(_beat(voice, v))
                    cursor += 64 // v
                nxt = events[i + 1].start if i + 1 < len(events) else b_end
                dur_units = _second_to_units(nxt - event.start, arrangement.tempo)
                v = _duration_code(dur_units)
                voice.beats.append(_note_beat(voice, event, v, len(tab_track.tuning)))
                cursor += 64 // v
            remaining = bar_units - cursor
            for v in _split_units(remaining):
                voice.beats.append(_beat(voice, v))
    _write(song, path)


def _beat(voice, value):
    beat = Beat(voice=voice)
    beat.duration = Duration(value)
    return beat


def _note_beat(voice, event, value, n_strings):
    beat = Beat(voice=voice)
    beat.duration = Duration(value)
    for n in event.notes:
        note = Note(beat=beat)
        note.string = n_strings - n.string
        note.value = n.fret
        beat.notes.append(note)
    return beat


def _write(song, path):
    import guitarpro

    guitarpro.write(song, path, version=(5, 1, 0))