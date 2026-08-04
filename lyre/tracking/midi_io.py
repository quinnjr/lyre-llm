import pretty_midi


def _ts(time_signature):
    num, den = time_signature
    return pretty_midi.TimeSignature(num, den, 0.0)


def notes_to_midi(notes, program=0, name="track", tempo=120.0, time_signature=(4, 4)):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    midi.time_signature_changes = [_ts(time_signature)]
    instrument = pretty_midi.Instrument(program=program, name=name)
    instrument.notes = [
        pretty_midi.Note(velocity=int(n.velocity), pitch=n.pitch, start=n.start, end=n.end)
        for n in notes
    ]
    midi.instruments.append(instrument)
    return midi


def write_multitrack(midi, path):
    midi.write(path)


def merge_stems(stem_notes, programs=None, tempo=120.0, time_signature=(4, 4)):
    programs = programs or {}
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    midi.time_signature_changes = [_ts(time_signature)]
    for stem, notes in stem_notes.items():
        instrument = pretty_midi.Instrument(
            program=programs.get(stem, 0), name=stem
        )
        instrument.notes = [
            pretty_midi.Note(velocity=int(n.velocity), pitch=n.pitch, start=n.start, end=n.end)
            for n in notes
        ]
        midi.instruments.append(instrument)
    return midi


def merge_instruments(instruments, tempo=120.0, time_signature=(4, 4)):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    midi.time_signature_changes = [_ts(time_signature)]
    for inst in instruments:
        instrument = pretty_midi.Instrument(
            program=inst.program, name=inst.name
        )
        instrument.is_drum = inst.is_drum
        instrument.notes = [
            pretty_midi.Note(velocity=int(n.velocity), pitch=n.pitch, start=n.start, end=n.end)
            for n in inst.notes
        ]
        midi.instruments.append(instrument)
    return midi
