"""Suite-wide guards and the shared arrangement fixtures.

The guards below make two properties that the suite relies on into enforced
facts rather than accidents of how the tests happen to be written: nothing
reaches the network, and no test can leave the global torch RNG advanced for
whichever test runs next.

The `make_*` helpers used to be copied into the individual test modules, where
they drifted apart -- three `_notes` with three different signatures, two
`_drum_notes` with different patterns, two `_arrangement` with different
defaults. Moving a test between modules then silently changed the music it was
asserting about, so there is one definition of each here.
"""

import socket

import pytest
import torch

from lyre.arranger.arrange import build_arrangement
from lyre.instruments import Instrument
from lyre.tracking.hmm import Note


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly rather than downloading anything.

    The separator builds its Demucs model lazily, so nothing downloads today --
    but one test calling transcribe() without a stub would quietly pull ~300MB
    of weights and take the suite offline-hostile with it. The socket guard is
    the backstop for everything else (the LLM labeller talks urllib); every
    test that exercises a request stubs the transport, so nothing legitimately
    needs a real socket.
    """
    import demucs.pretrained

    import lyre.separator.demucs as separator_module

    def _no_weights(*args, **kwargs):
        pytest.fail("test tried to download Demucs weights")

    monkeypatch.setattr(demucs.pretrained, "get_model", _no_weights)
    # The separator module imported get_model by name, so patching only
    # demucs.pretrained would leave the real one reachable.
    monkeypatch.setattr(separator_module, "get_model", _no_weights)
    monkeypatch.setattr(
        socket, "socket", lambda *a, **k: pytest.fail("test opened a socket")
    )


@pytest.fixture(autouse=True)
def _isolate_rng():
    """Restore the global torch RNG so seeding in one test cannot leak."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


# --------------------------------------------------------------------------
# shared note / arrangement builders
# --------------------------------------------------------------------------

GUITAR_PITCHES = [40, 45, 50, 55, 59, 64, 52, 57]
BASS_PITCHES = [28, 33, 38, 43, 30, 35, 40, 45]

# kick, closed hat, snare, closed hat -- the smallest pattern that produces all
# three of the drum parts the renderers lay out on separate rows.
DRUM_PATTERN = [36, 42, 38, 42]


def make_notes(pitches, step=0.25, start=0.0):
    """Back-to-back notes, one per pitch, each `step` seconds long."""
    return [
        Note(
            pitch=p,
            start=start + i * step,
            end=start + (i + 1) * step,
            velocity=100.0,
        )
        for i, p in enumerate(pitches)
    ]


def make_drum_notes(count=24, step=0.25, start=0.0):
    """`count` short drum hits cycling through DRUM_PATTERN."""
    return [
        Note(
            pitch=DRUM_PATTERN[i % len(DRUM_PATTERN)],
            start=start + i * step,
            end=start + i * step + 0.1,
            velocity=100.0,
        )
        for i in range(count)
    ]


def make_arrangement(
    ts=(4, 4),
    drums=True,
    start=0.0,
    step=0.25,
    guitar_pitches=None,
    bass_pitches=None,
    drum_count=24,
    drum_step=None,
    guitar_tuning="standard",
    bass_tuning="standard",
    tempo=120.0,
):
    """A guitar + bass (+ drums) arrangement at 120 bpm.

    The defaults give several bars in every time signature under test; a
    single-bar fixture makes "only in measure 1" assertions vacuous.
    """
    if drum_step is None:
        drum_step = step
    guitar = GUITAR_PITCHES * 3 if guitar_pitches is None else guitar_pitches
    bass = BASS_PITCHES * 3 if bass_pitches is None else bass_pitches
    instruments = [
        Instrument(name="guitar", notes=make_notes(guitar, step=step, start=start)),
        Instrument(name="bass", notes=make_notes(bass, step=step, start=start)),
    ]
    if drums:
        instruments.append(
            Instrument(
                name="drums",
                notes=make_drum_notes(drum_count, step=drum_step, start=start),
            )
        )
    return build_arrangement(
        instruments,
        tempo=tempo,
        time_signature=ts,
        guitar_tuning=guitar_tuning,
        bass_tuning=bass_tuning,
    )
