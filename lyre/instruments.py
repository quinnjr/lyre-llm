from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

from lyre.tracking.hmm import Note

FAMILY_GUITAR = "guitar"
FAMILY_BASS = "bass"
FAMILY_DRUMS = "drums"
FAMILY_KEYS = "keys"
FAMILY_VOCALS = "vocals"
FAMILY_STRINGS = "strings"
FAMILY_SYNTH = "synth"
FAMILY_OTHER = "other"

FAMILIES = [
    FAMILY_GUITAR,
    FAMILY_BASS,
    FAMILY_DRUMS,
    FAMILY_KEYS,
    FAMILY_VOCALS,
    FAMILY_STRINGS,
    FAMILY_SYNTH,
    FAMILY_OTHER,
]

LABEL_VOCAB = [
    "guitar rhythm",
    "guitar lead",
    "guitar",
    "bass",
    "drums",
    "piano",
    "vocals",
    "strings",
    "synth",
    "other",
]

_LABEL_FAMILY = {
    "guitar rhythm": FAMILY_GUITAR,
    "guitar lead": FAMILY_GUITAR,
    "guitar": FAMILY_GUITAR,
    "acoustic guitar": FAMILY_GUITAR,
    "electric guitar": FAMILY_GUITAR,
    "bass": FAMILY_BASS,
    "bass guitar": FAMILY_BASS,
    "drums": FAMILY_DRUMS,
    "percussion": FAMILY_DRUMS,
    "piano": FAMILY_KEYS,
    "keys": FAMILY_KEYS,
    "keyboard": FAMILY_KEYS,
    "vocals": FAMILY_VOCALS,
    "voice": FAMILY_VOCALS,
    "strings": FAMILY_STRINGS,
    "synth": FAMILY_SYNTH,
    "synthesizer": FAMILY_SYNTH,
    "other": FAMILY_OTHER,
}

_LABEL_KEYS_BY_LENGTH = sorted(_LABEL_FAMILY, key=len, reverse=True)

FAMILY_PROGRAMS = {
    FAMILY_GUITAR: 30,
    FAMILY_BASS: 33,
    FAMILY_KEYS: 0,
    FAMILY_VOCALS: 54,
    FAMILY_STRINGS: 48,
    FAMILY_SYNTH: 80,
    FAMILY_OTHER: 0,
}


def family_of_label(label):
    return _LABEL_FAMILY.get(str(label).strip().lower(), FAMILY_OTHER)


def program_for(label):
    return FAMILY_PROGRAMS.get(family_of_label(label), 0)


def normalize_label(label):
    label = str(label).strip().lower()
    if label in _LABEL_FAMILY:
        return label
    for word in _LABEL_KEYS_BY_LENGTH:
        if word in label:
            return word
    return "other"


@dataclass
class Instrument:
    name: str
    notes: list = field(default_factory=list)
    program: int = 0
    is_drum: bool = False
    source: str = ""

    @property
    def family(self):
        return family_of_label(self.name)

    def features(self):
        notes = self.notes
        if not notes:
            return {
                "n_notes": 0,
                "mean_polyphony": 0.0,
                "chordal_fraction": 0.0,
                "pitch_min": 0,
                "pitch_max": 0,
                "duration": 0.0,
                "single_line_fraction": 0.0,
            }
        pitches = [n.pitch for n in notes]
        onsets = {}
        for n in notes:
            onsets.setdefault(round(n.start, 2), []).append(n)
        poly = [len(g) for g in onsets.values()]
        mean_poly = sum(poly) / len(poly)
        chordal = sum(1 for p in poly if p >= 3) / len(poly)
        single = sum(1 for p in poly if p == 1) / len(poly)
        return {
            "n_notes": len(notes),
            "mean_polyphony": round(mean_poly, 3),
            "chordal_fraction": round(chordal, 3),
            "single_line_fraction": round(single, 3),
            "pitch_min": min(pitches),
            "pitch_max": max(pitches),
            "duration": round(max(n.end for n in notes), 3),
        }


def _overlap_counts(notes, eps=1e-3):
    """Number of other notes overlapping each note, in O(n log n).

    ``other`` overlaps ``note`` when ``other.end > note.start + eps`` and
    ``other.start < note.end - eps``. Counting is done with two sorted arrays:
    (notes starting before the note ends) minus (notes ending before it starts).
    Degenerate notes shorter than ``2 * eps`` break that subtraction, so they
    fall back to a direct scan.
    """
    starts = sorted(n.start for n in notes)
    ends = sorted(n.end for n in notes)
    counts = []
    for note in notes:
        hi = note.end - eps
        lo = note.start + eps
        if lo < hi:
            count = bisect_left(starts, hi) - bisect_right(ends, lo)
            # the note itself is counted among the starts, never among the ends
            counts.append(count - 1)
        else:
            counts.append(
                sum(
                    1
                    for other in notes
                    if other is not note
                    and other.end > lo
                    and other.start < hi
                )
            )
    return counts


def split_guitar(instrument):
    rhythm, lead = [], []
    overlaps = _overlap_counts(instrument.notes)
    for note, overlap in zip(instrument.notes, overlaps):
        if overlap >= 2:
            rhythm.append(note)
        else:
            lead.append(note)
    return rhythm, lead
