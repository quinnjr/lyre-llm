"""Bar-bucketing and drum-grouping shared by the ascii, GP5 and MusicXML renderers.

These used to be copied into each renderer. The copies have to agree exactly --
if GP5 and MusicXML disagree about how many drum bars there are, the two exports
of the same arrangement no longer line up -- so there is one copy here.
"""

from ..arrange import bar_seconds


# Drum hit start times are quantised to this many decimal places before they
# are grouped or counted. Both operations must use it: keying events on a
# rounded start while sizing the bar list from the raw start lets a hit that
# rounds up onto a bar line land past the last bar and disappear.
DRUM_TIME_PLACES = 4


def bucket_by_bar(events, bars):
    """Bucket events into bars with a single pass over the sorted events."""
    buckets = [[] for _ in bars]
    ordered = sorted(events, key=lambda e: e.start)
    i = 0
    last = len(bars) - 1
    for b_idx, (b_start, b_end) in enumerate(bars):
        while i < len(ordered) and ordered[i].start < b_start:
            i += 1
        # The final bar is closed at the top so floating-point start times
        # slightly past its nominal end cannot fall off the end of the piece.
        hi = float("inf") if b_idx == last else b_end
        while i < len(ordered) and ordered[i].start < hi:
            buckets[b_idx].append(ordered[i])
            i += 1
    return buckets


def drum_bars(hits, tempo, ts):
    """Bar spans covering every drum hit, as (start_sec, end_sec) pairs."""
    if not hits:
        return []
    bar_sec = bar_seconds(tempo, ts)
    latest = max(round(h.start, DRUM_TIME_PLACES) for h in hits)
    n_bars = int(latest / bar_sec) + 1
    return [(i * bar_sec, (i + 1) * bar_sec) for i in range(n_bars)]


class DrumEvent:
    """A TabEvent-lookalike: a start time plus the simultaneous drum payloads."""

    __slots__ = ("start", "values")

    def __init__(self, start, values):
        self.start = start
        self.values = values


def drum_events(hits, payload, sort_key=None):
    """Collapse simultaneous drum hits into DrumEvent records.

    `payload(hit)` yields whatever the caller wants stored -- a MIDI key for
    GP5, a part name for MusicXML -- and returning None drops the hit.
    `sort_key`, when given, orders the payloads within each event; otherwise
    they keep first-seen order.
    """
    grouped = {}
    for h in hits:
        key = round(h.start, DRUM_TIME_PLACES)
        values = grouped.setdefault(key, [])
        value = payload(h)
        if value is not None and value not in values:
            values.append(value)
    out = []
    for key in sorted(grouped):
        values = grouped[key]
        if not values:
            continue
        if sort_key is not None:
            values = sorted(values, key=sort_key)
        out.append(DrumEvent(start=key, values=values))
    return out
