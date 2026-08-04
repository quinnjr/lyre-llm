"""Audio decode / channel-fit and MIDI unit tests.

Everything here is a *units* test: the failure mode being guarded against is
output that has the right shape and is silently wrong (a stereo file narrowed to
mono by dropping a channel, a note written at velocity 0 so it is inaudible).
"""

import inspect

import pretty_midi
import pytest
import torch

from lyre.errors import AudioDecodeError
from lyre.io.decode import _fit_channels, load_audio, save_wav
from lyre.tracking import midi_io
from lyre.tracking.hmm import Note
from lyre.instruments import Instrument


# ---------------------------------------------------------------- _fit_channels


def test_fit_channels_stereo_to_stereo_is_untouched():
    wav = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    out = _fit_channels(wav, 2)
    assert out.shape == (2, 3)
    assert torch.equal(out, wav)


def test_fit_channels_stereo_to_mono_is_the_exact_mean():
    # Regression: dropping channel 1 instead of averaging loses everything
    # panned hard right, and the waveform still has the expected shape.
    wav = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    out = _fit_channels(wav, 1)
    assert out.shape == (1, 3)
    assert torch.allclose(out, torch.tensor([[5.0, 6.0, 7.0]]))


@pytest.mark.parametrize("have", [2, 3, 6, 8])
def test_fit_channels_to_mono_is_exactly_the_channel_mean(have):
    # Mono is the one width with an unambiguous answer, and every other width
    # must agree with it: the surplus-folding branch below is a generalisation
    # of the mean, not a different rule.
    torch.manual_seed(0)
    wav = torch.rand(have, 16)
    out = _fit_channels(wav, 1)
    assert out.shape == (1, 16)
    assert torch.equal(out[0], wav.mean(dim=0))


def test_fit_channels_mono_to_stereo_duplicates_the_row():
    wav = torch.tensor([[1.0, 2.0, 3.0]])
    out = _fit_channels(wav, 2)
    assert out.shape == (2, 3)
    assert torch.equal(out[0], out[1])
    assert torch.equal(out[0], wav[0])


def test_fit_channels_six_to_two_folds_the_surplus_in():
    # Six constant rows 1..6. Each output channel is a convex combination of
    # all six inputs: its own with weight channels/have = 2/6, each of the four
    # surplus channels with weight 1/6. out[0] = (2*1 + 3+4+5+6) / 6 = 10/3.
    wav = torch.stack([torch.full((4,), float(i)) for i in range(1, 7)])
    out = _fit_channels(wav, 2)
    assert out.shape == (2, 4)
    assert torch.allclose(out[0], torch.full((4,), 10.0 / 3.0))
    assert torch.allclose(out[1], torch.full((4,), 11.0 / 3.0))


def test_fit_channels_six_to_two_constant_input():
    # Identical content on every channel maps to itself: the same material
    # delivered as 5.1 and as stereo has to produce the same loudness, and so
    # the same log-mel. Weights that did not sum to 1 made this 1.5x too loud.
    wav = torch.full((6, 5), 2.0)
    out = _fit_channels(wav, 2)
    assert torch.allclose(out, torch.full((2, 5), 2.0))


@pytest.mark.parametrize("have,channels", [(3, 2), (6, 2), (8, 2), (6, 4), (5, 3)])
def test_fit_channels_downmix_never_exceeds_the_input_peak(have, channels):
    # Full-scale input: adding a fraction of the surplus on top of full-
    # amplitude kept channels clipped on re-encode. A convex combination cannot.
    wav = torch.full((have, 8), 1.0)
    out = _fit_channels(wav, channels)
    assert out.shape == (channels, 8)
    assert float(out.max()) <= 1.0

    torch.manual_seed(1)
    wav = torch.rand(have, 64) * 2.0 - 1.0
    out = _fit_channels(wav, channels)
    assert float(out.abs().max()) <= float(wav.abs().max()) + 1e-6


def test_fit_channels_downmix_weights_sum_to_one():
    # Stated directly: feeding a unit impulse down every channel in turn and
    # summing the responses must give exactly 1 in each output channel.
    have, channels = 6, 2
    total = torch.zeros(channels, 1)
    for c in range(have):
        wav = torch.zeros(have, 1)
        wav[c, 0] = 1.0
        total += _fit_channels(wav, channels)
    assert torch.allclose(total, torch.ones(channels, 1))


def test_fit_channels_none_passes_through():
    wav = torch.zeros(3, 7)
    assert _fit_channels(wav, None).shape == (3, 7)


# ------------------------------------------------------------------- load_audio


def test_load_audio_on_a_non_audio_file_raises_audio_decode_error(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("this is definitely not a wav file\n")
    with pytest.raises(AudioDecodeError) as exc:
        load_audio(str(junk))
    assert str(junk) in str(exc.value)


def test_load_audio_upmixes_a_mono_file_to_stereo(tmp_path):
    path = str(tmp_path / "mono.wav")
    samples = torch.linspace(-0.5, 0.5, 800).unsqueeze(0)
    save_wav(path, samples, 8000)

    wav, sr = load_audio(path, sample_rate=8000, channels=2)
    assert sr == 8000
    assert wav.shape[0] == 2
    assert torch.equal(wav[0], wav[1])
    assert float(wav.abs().max()) > 0.1


def test_audio_info_is_gone():
    # Deleted API: it reported metadata that nothing checked against the decoded
    # waveform, so callers trusted a rate the decode never produced.
    import lyre.io
    import lyre.io.decode

    assert not hasattr(lyre.io, "audio_info")
    assert not hasattr(lyre.io.decode, "audio_info")
    assert "audio_info" not in lyre.io.__all__
    with pytest.raises(ImportError):
        from lyre.io import audio_info  # noqa: F401


def test_load_audio_has_no_offset_or_duration_parameters():
    params = inspect.signature(load_audio).parameters
    assert "offset_sec" not in params
    assert "duration_sec" not in params
    assert list(params) == ["path", "sample_rate", "channels"]


# --------------------------------------------------------------------- velocity


@pytest.mark.parametrize(
    "velocity,expected",
    [
        (0.0, 1),
        (0.4, 1),
        (63.5, 64),
        (126.6, 127),
        (500.0, 127),
        (-5.0, 1),
    ],
)
def test_velocity_clamps_into_the_audible_range(velocity, expected):
    note = Note(pitch=60, start=0.0, end=1.0, velocity=velocity)
    assert midi_io._velocity(note) == expected


def test_merge_instruments_round_trips_through_pretty_midi(tmp_path):
    guitar = Instrument(
        name="guitar",
        notes=[
            Note(pitch=64, start=0.0, end=0.5, velocity=0.3),
            Note(pitch=67, start=0.5, end=1.0, velocity=90.0),
            Note(pitch=71, start=1.0, end=1.5, velocity=400.0),
        ],
        program=27,
        source="guitar",
    )
    drums = Instrument(
        name="drums",
        notes=[Note(pitch=36, start=0.0, end=0.1, velocity=100.0)],
        program=0,
        is_drum=True,
        source="drums",
    )

    path = str(tmp_path / "score.mid")
    midi_io.merge_instruments([guitar, drums], tempo=120.0, time_signature=(4, 4))
    midi_io.merge_instruments([guitar, drums]).write(path)

    back = pretty_midi.PrettyMIDI(path)
    by_name = {i.name: i for i in back.instruments}
    assert set(by_name) == {"guitar", "drums"}

    every = [n for i in back.instruments for n in i.notes]
    assert every, "round trip produced no notes at all"
    assert all(1 <= n.velocity <= 127 for n in every)

    # velocity=0.3 must not round to 0: a MIDI note-on with velocity 0 is a
    # note-off, so the note would be written and be completely inaudible.
    quiet = [n for n in by_name["guitar"].notes if n.pitch == 64]
    assert len(quiet) == 1
    assert quiet[0].velocity == 1

    loud = [n for n in by_name["guitar"].notes if n.pitch == 71]
    assert loud[0].velocity == 127

    assert by_name["drums"].is_drum is True
    assert by_name["guitar"].is_drum is False
