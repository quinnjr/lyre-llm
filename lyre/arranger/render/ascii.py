from lyre.arranger.arrange import string_label


def _string_labels(track):
    labels = [string_label(p) for p in track.tuning]
    labels[0] = labels[0].lower()
    labels[-1] = labels[-1].upper()
    return labels[::-1]


def _render_tab_track(track, labels, cols_per_bar=32):
    if not track.bars:
        return f"[{track.name}: no content]"
    lines = [f"{track.name} ({len(track.bars)} bars)"]
    for b_start, b_end in track.bars:
        bar_events = [e for e in track.events if b_start <= e.start < b_end]
        duration = b_end - b_start
        col_for = {}
        for e in bar_events:
            col = min(cols_per_bar - 1, int((e.start - b_start) / duration * cols_per_bar))
            col_for.setdefault(col, []).append(e)
        grid = [["-"] * cols_per_bar for _ in track.tuning]
        for col, events in col_for.items():
            for e in events:
                for n in e.notes:
                    label_idx = len(track.tuning) - 1 - n.string
                    grid[label_idx][col] = str(n.fret)
        rendered = [f"{label}|" + "".join(row) + "|" for label, row in zip(labels, grid)]
        lines.extend(rendered)
        lines.append("")
    return "\n".join(lines)


def _render_drums(hits, tempo, ts, cols_per_bar=32):
    if not hits:
        return "[Drums: no content]"
    from lyre.arranger.arrange import bar_seconds

    parts = ["kick", "snare", "hat", "crash", "ride", "tom"]
    lines = ["Drums"]
    bar_sec = bar_seconds(tempo, ts)
    n_bars = int(max(h.start for h in hits) / bar_sec) + 1
    for part in parts:
        cells = ["."] * (n_bars * cols_per_bar)
        for h in hits:
            if h.part != part:
                continue
            global_col = int(h.start / (bar_sec / cols_per_bar))
            if global_col < len(cells):
                cells[global_col] = "x"
        chunks = ["".join(cells[i * cols_per_bar : (i + 1) * cols_per_bar]) for i in range(n_bars)]
        lines.append(f"{part:>5}: " + "|".join(chunks))
    return "\n".join(lines)


def render_all_ascii(arrangement):
    parts = []
    for track in arrangement.guitar:
        parts.append(_render_tab_track(track, _string_labels(track)))
        parts.append("")
    for track in arrangement.bass:
        parts.append(_render_tab_track(track, _string_labels(track)))
        parts.append("")
    parts.append(_render_drums(arrangement.drums, arrangement.tempo, arrangement.ts))
    return "\n\n".join(parts)
