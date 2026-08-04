# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import numpy as np
import pretty_midi


def midi_to_frames(
    midi_path,
    duration=None,
    n_notes=128,
    min_note=24,
    max_note=95,
    fps=100,
):
    """Rasterise a MIDI file onto a frame grid.

    Returns a dict with:
      ``pitch``    (n_frames, n_notes) float32 sustain map, 0/1.
      ``onset``    (n_frames,)         float32 per-frame onset indicator (any pitch).
                   Deliberately a per-frame scalar, not per-pitch: the model's
                   onset head predicts a single curve and the tracker consumes
                   one. Widening it here would silently break both.
      ``velocity`` (n_frames, n_notes) float32 per-note velocity, scaled 0..1.
      ``duration`` the duration in seconds the grid covers.

    Notes that start at or after ``duration`` are skipped; notes that straddle
    the end are truncated.
    """
    # Validate the range once here so the per-note loop needs a single guard;
    # an out-of-range max_note would otherwise index past the pitch axis.
    if not (0 <= min_note <= max_note < n_notes):
        raise ValueError(
            "midi_to_frames requires 0 <= min_note <= max_note < n_notes; got "
            "min_note=%r, max_note=%r, n_notes=%r" % (min_note, max_note, n_notes)
        )
    midi = pretty_midi.PrettyMIDI(midi_path)
    if duration is None:
        duration = midi.get_end_time()
    n_frames = int(np.ceil(duration * fps)) + 1
    pitch = np.zeros((n_frames, n_notes), dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    velocity = np.zeros((n_frames, n_notes), dtype=np.float32)
    for instrument in midi.instruments:
        for note in instrument.notes:
            if not (min_note <= note.pitch <= max_note):
                continue
            start = int(note.start * fps)
            if start >= n_frames:
                continue
            start = max(0, start)
            end = max(start + 1, int(note.end * fps))
            end = min(end, n_frames)
            pitch[start:end, note.pitch] = 1.0
            onset[start] = 1.0
            velocity[start:end, note.pitch] = np.maximum(
                velocity[start:end, note.pitch], note.velocity / 127.0
            )
    return {
        "pitch": pitch,
        "onset": onset,
        "velocity": velocity,
        "duration": duration,
    }
