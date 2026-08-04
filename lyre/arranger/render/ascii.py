from collections import defaultdict

from lyre.arranger.arrange import string_label

from ._common import DRUM_TIME_PLACES, bucket_by_bar, drum_bars


def _string_labels(track):
    labels = [string_label(p) for p in track.tuning]
    labels[0] = labels[0].lower()
    labels[-1] = labels[-1].upper()
    return labels[::-1]


def _render_tab_track(track, labels, cols_per_bar=32):
    if not track.bars:
        return f"[{track.name}: no content]"
    lines = [f"{track.name} ({len(track.bars)} bars)"]
    buckets = bucket_by_bar(track.events, track.bars)
    for bar_idx, (b_start, b_end) in enumerate(track.bars):
        bar_events = buckets[bar_idx]
        duration = b_end - b_start
        col_for = {}
        for e in bar_events:
            col = min(cols_per_bar - 1, int((e.start - b_start) / duration * cols_per_bar))
            col_for.setdefault(col, []).append(e)
        # Every cell holds a fixed-width token so two-digit frets cannot push a
        # row out of alignment with its neighbours.
        grid = [[None] * cols_per_bar for _ in track.tuning]
        for col, events in col_for.items():
            for e in events:
                for n in e.notes:
                    label_idx = len(track.tuning) - 1 - n.string
                    grid[label_idx][col] = str(n.fret)
        widths = [
            max([1] + [len(grid[r][c]) for r in range(len(grid)) if grid[r][c]])
            for c in range(cols_per_bar)
        ]
        rendered = []
        for label, row in zip(labels, grid):
            cells = [
                (cell if cell is not None else "").rjust(w, "-")
                for cell, w in zip(row, widths)
            ]
            rendered.append(f"{label}|" + "".join(cells) + "|")
        lines.extend(rendered)
        lines.append("")
    return "\n".join(lines)


def _render_drums(hits, tempo, ts, cols_per_bar=32):
    if not hits:
        return "[Drums: no content]"
    parts = ["kick", "snare", "hat", "crash", "ride", "tom"]
    lines = ["Drums"]
    bars = drum_bars(hits, tempo, ts)
    n_bars = len(bars)
    bar_sec = bars[0][1] - bars[0][0]
    # One pass over the hits instead of one pass per part.
    by_part = defaultdict(list)
    for h in hits:
        by_part[h.part].append(h)
    n_cells = n_bars * cols_per_bar
    for part in parts:
        cells = ["."] * n_cells
        for h in by_part.get(part, ()):
            # Quantise exactly as drum_bars/drum_events do. Dividing the raw
            # start instead puts a hit a hair under a bar line in the previous
            # bar's last cell, while GP5 and MusicXML round it onto the downbeat
            # -- one arrangement, three exports disagreeing about the bar.
            start = round(h.start, DRUM_TIME_PLACES)
            global_col = int(start / (bar_sec / cols_per_bar))
            if global_col < n_cells:
                cells[global_col] = "x"
        chunks = ["".join(cells[i * cols_per_bar : (i + 1) * cols_per_bar]) for i in range(n_bars)]
        lines.append(f"{part:>5}: " + "|".join(chunks))
    return "\n".join(lines)


def _render_notes(track):
    """Render the arranger's human-readable notes for a track, when present."""
    if not track.performance_notes:
        return None
    lines = [f"{track.name} notes:"]
    lines.extend(f"  - {n}" for n in track.performance_notes)
    return "\n".join(lines)


def _render_group(tracks):
    """Blocks for a group of tab tracks: the tab itself plus any notes."""
    blocks = []
    for track in tracks:
        blocks.append(_render_tab_track(track, _string_labels(track)))
        notes = _render_notes(track)
        if notes:
            blocks.append(notes)
        blocks.append("")
    return blocks


def render_all_ascii(arrangement):
    parts = _render_group(list(arrangement.guitar) + list(arrangement.bass))
    parts.append(_render_drums(arrangement.drums, arrangement.tempo, arrangement.ts))
    return "\n\n".join(parts)


def render_parts(arrangement):
    """Return (guitar_text, bass_text, drums_text) as three independent renders.

    Grouping keys on the track objects the arranger placed in `arrangement.guitar`
    / `.bass` / `.drums`, never on track names -- a user is free to call a guitar
    track "Bass Guitar", and the split must still follow the arrangement.
    """
    guitar = "\n\n".join(_render_group(list(arrangement.guitar))).rstrip()
    bass = "\n\n".join(_render_group(list(arrangement.bass))).rstrip()
    drums = _render_drums(arrangement.drums, arrangement.tempo, arrangement.ts)
    return guitar, bass, drums
