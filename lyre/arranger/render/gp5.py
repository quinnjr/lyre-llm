from guitarpro import Song, Track, Beat, Note, Duration, TimeSignature
from guitarpro import GuitarString, MidiChannel, BeatStatus

from ..arrange import GUITAR_TUNING
from ._common import bucket_by_bar, drum_bars, drum_events

# One whole note is UNITS_PER_WHOLE units, so a quarter note is 16 units and a
# guitarpro Duration with `value` v spans UNITS_PER_WHOLE // v units.
UNITS_PER_WHOLE = 64

_DURATIONS = [1, 2, 4, 8, 16, 32, 64]

PERCUSSION_CHANNEL = 9
PERCUSSION_TUNING = [40, 45, 50, 55, 59, 64]

# Representative General MIDI percussion key for each classified drum part.
DRUM_MIDI = {
    "kick": 36,
    "snare": 38,
    "hat": 42,
    "crash": 49,
    "ride": 51,
    "tom": 45,
}


def _bar_units(numerator, denominator):
    return numerator * UNITS_PER_WHOLE // denominator


def _duration_code(units):
    for v in _DURATIONS:
        if UNITS_PER_WHOLE // v <= units:
            return v
    return 64


def _duration_units(value):
    return UNITS_PER_WHOLE // value


def _split_units(units):
    chosen = []
    guard = 0
    while units > 0:
        v = _duration_code(units)
        step = _duration_units(v)
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
    return int(round(quarters * UNITS_PER_WHOLE / 4))


def _second_to_units(sec, tempo):
    return _quarters_to_units(_note_to_quarters(sec, tempo))


def write_gp5(arrangement, path):
    numerator, denominator = arrangement.ts
    song = Song(title="lyre transcription", tempo=int(round(arrangement.tempo)))
    time_sig = TimeSignature(
        numerator=numerator,
        denominator=Duration(denominator),
    )
    all_tracks = list(arrangement.guitar) + list(arrangement.bass)
    tracks_meta = [(t.name, t.tuning, t.max_fret, False) for t in all_tracks]
    drum_hits = list(arrangement.drums)
    # A drum-free arrangement must not gain a phantom percussion track whose
    # measures would round-trip with zero beats.
    if drum_hits:
        tracks_meta.append(("Drums", PERCUSSION_TUNING, 24, True))
    if not tracks_meta:
        tracks_meta = [("empty", GUITAR_TUNING, 24, False)]
    # Song() ships with a default track and header, and Track(song=...) seeds one
    # Measure per existing header. Clear the headers first so every track starts
    # empty and measure counts stay in step with measureHeaders.
    song.measureHeaders = []
    built = []
    for idx, (name, tuning, max_fret, is_percussion) in enumerate(tracks_meta):
        track = Track(song=song, number=idx + 1, isPercussionTrack=is_percussion)
        track.name = name
        track.fretCount = max_fret
        if is_percussion:
            track.channel = MidiChannel(
                channel=PERCUSSION_CHANNEL,
                effectChannel=PERCUSSION_CHANNEL,
                instrument=0,
            )
        # Guitar Pro numbers strings 1..n from the highest-pitched string down,
        # while our tunings are stored low-to-high.
        track.strings = [
            GuitarString(number=i + 1, value=p)
            for i, p in enumerate(reversed(tuning))
        ]
        built.append(track)
    song.tracks = built
    d_bars = drum_bars(drum_hits, arrangement.tempo, arrangement.ts)
    n_bars = max(
        [len(t.bars) for t in all_tracks] + [len(d_bars)], default=1
    ) or 1
    bar_units = _bar_units(numerator, denominator)
    for i in range(n_bars):
        song.newMeasure()
        header = song.measureHeaders[i]
        header.timeSignature = time_sig
        header.number = i + 1
    for g_idx, tab_track in enumerate(all_tracks):
        buckets = bucket_by_bar(tab_track.events, tab_track.bars)
        for bar_idx, (b_start, b_end) in enumerate(tab_track.bars):
            measure = song.tracks[g_idx].measures[bar_idx]
            _fill_voice(
                measure,
                buckets[bar_idx],
                b_start,
                b_end,
                bar_units,
                arrangement.tempo,
                n_strings=len(tab_track.tuning),
            )
        _pad_measures(song.tracks[g_idx], len(tab_track.bars), n_bars, bar_units)
    if drum_hits and d_bars:
        d_idx = len(all_tracks)
        events = drum_events(drum_hits, lambda h: DRUM_MIDI.get(h.part))
        buckets = bucket_by_bar(events, d_bars)
        for bar_idx, (b_start, b_end) in enumerate(d_bars):
            measure = song.tracks[d_idx].measures[bar_idx]
            _fill_voice(
                measure,
                buckets[bar_idx],
                b_start,
                b_end,
                bar_units,
                arrangement.tempo,
            )
        _pad_measures(song.tracks[d_idx], len(d_bars), n_bars, bar_units)
    if not all_tracks and not drum_hits:
        # The "empty" placeholder track has no bars of its own, so nothing above
        # filled it. Rest it out, or its measures round-trip with zero beats.
        _pad_measures(song.tracks[0], 0, n_bars, bar_units)
    _write(song, path)


def _pad_measures(track, first, last, bar_units):
    """Fill trailing measures a track does not reach with full-bar rests."""
    for bar_idx in range(first, last):
        voice = track.measures[bar_idx].voices[0]
        voice.beats = []
        for v in _split_units(bar_units):
            voice.beats.append(_beat(voice, v))


def _fill_voice(measure, events, b_start, b_end, bar_units, tempo, n_strings=None):
    """Fill one measure. `n_strings` selects tab notes; omit it for drum events."""
    # GP5 always stores Measure.maxVoices voices; replacing the list with a
    # single voice desynchronises the reader (it fails on "voice 2").
    voice = measure.voices[0]
    voice.beats = []
    cursor = 0
    last_beat = None
    for i, event in enumerate(events):
        ev_start = _second_to_units(event.start - b_start, tempo)
        gap = ev_start - cursor
        for v in _split_units(gap):
            voice.beats.append(_beat(voice, v))
            cursor += _duration_units(v)
        nxt = events[i + 1].start if i + 1 < len(events) else b_end
        dur_units = _second_to_units(nxt - event.start, tempo)
        dur_units = min(dur_units, bar_units - cursor)
        if dur_units <= 0:
            # Zero-length event (two onsets inside the chord tolerance): fold it
            # into the previous beat instead of emitting a phantom 64th that
            # would advance the cursor and desynchronise the bar.
            if last_beat is not None:
                if n_strings is None:
                    _merge_drum_event(last_beat, event)
                else:
                    _merge_tab_event(last_beat, event, n_strings)
            continue
        v = _duration_code(dur_units)
        if n_strings is None:
            beat = _drum_beat(voice, event, v)
        else:
            beat = _note_beat(voice, event, v, n_strings)
        voice.beats.append(beat)
        last_beat = beat
        cursor += _duration_units(v)
    remaining = bar_units - cursor
    for v in _split_units(remaining):
        voice.beats.append(_beat(voice, v))


def _beat(voice, value):
    """A rest beat. Status must be explicit: a beat left at BeatStatus.empty is
    read back as zero-length, which collapses every beat in the measure."""
    beat = Beat(voice=voice)
    beat.duration = Duration(value)
    beat.status = BeatStatus.rest
    return beat


def _note_beat(voice, event, value, n_strings):
    beat = Beat(voice=voice)
    beat.duration = Duration(value)
    beat.status = BeatStatus.normal
    _merge_tab_event(beat, event, n_strings)
    return beat


def _merge_tab_event(beat, event, n_strings):
    for n in event.notes:
        string = n_strings - n.string
        if any(existing.string == string for existing in beat.notes):
            continue
        note = Note(beat=beat)
        note.string = string
        note.value = n.fret
        beat.notes.append(note)


def _drum_beat(voice, event, value):
    beat = Beat(voice=voice)
    beat.duration = Duration(value)
    beat.status = BeatStatus.normal
    _merge_drum_event(beat, event)
    return beat


def _merge_drum_event(beat, event):
    used = {n.string for n in beat.notes}
    for midi in event.values:
        string = 1
        while string in used:
            string += 1
        if string > 6:
            break
        used.add(string)
        note = Note(beat=beat)
        note.string = string
        note.value = midi
        beat.notes.append(note)


def _write(song, path):
    import guitarpro

    guitarpro.write(song, path, version=(5, 1, 0))
