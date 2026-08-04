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
    midi = pretty_midi.PrettyMIDI(midi_path)
    if duration is None:
        duration = midi.get_end_time()
    n_frames = int(np.ceil(duration * fps)) + 1
    pitch = np.zeros((n_frames, n_notes), dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    velocity = np.zeros(n_frames, dtype=np.float32)
    for instrument in midi.instruments:
        for note in instrument.notes:
            if not (min_note <= note.pitch <= max_note):
                continue
            start = int(note.start * fps)
            end = max(start + 1, int(note.end * fps))
            end = min(end, n_frames)
            pitch[start:end, note.pitch] = 1.0
            onset[start] = 1.0
            velocity[start:end] = np.maximum(
                velocity[start:end], note.velocity / 127.0
            )
    return {
        "pitch": pitch,
        "onset": onset,
        "velocity": velocity,
        "duration": duration,
    }
