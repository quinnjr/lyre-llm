import os

import pytest
import torch

from lyre import errors
from lyre import pipeline as pipeline_mod
from lyre.arranger.arrange import (
    Arrangement,
    build_bass_track,
    build_guitar_track,
    classify_drums,
)
from lyre.errors import (
    AudioDecodeError,
    CheckpointNotFound,
    LyreError,
    PdfUnavailable,
)
from lyre.io.decode import save_wav
from lyre.pipeline import Converter, _shift_notes, _write_ascii_bundle, render_parts
from lyre.tracking.hmm import Note


def _notes(pitches, step=0.25):
    return [
        Note(pitch=p, start=i * step, end=(i + 1) * step, velocity=100.0)
        for i, p in enumerate(pitches)
    ]


def _adversarial_arrangement():
    """Track names chosen so substring matching on the name cannot work: the
    guitar track is called "Bass Guitar" and the bass track "Guitar (low)"."""
    guitar = build_guitar_track(
        _notes([64, 59, 55, 50, 45, 40, 52, 57]), 120.0, (4, 4), name="Bass Guitar"
    )
    bass = build_bass_track(
        _notes([28, 33, 38, 43, 30, 35, 40, 45]), 120.0, (4, 4), name="Guitar (low)"
    )
    drums = classify_drums(
        [
            Note(pitch=36, start=0.0, end=0.1, velocity=100.0),
            Note(pitch=38, start=0.5, end=0.6, velocity=100.0),
            Note(pitch=42, start=1.0, end=1.1, velocity=100.0),
        ]
    )
    return Arrangement(tempo=120.0, ts=(4, 4), guitar=[guitar], bass=[bass], drums=drums)


def test_render_parts_separates_guitar_from_bass():
    arrangement = _adversarial_arrangement()
    guitar_text, bass_text, drums_text = render_parts(arrangement)

    assert guitar_text.strip()
    assert bass_text.strip()
    assert drums_text.strip()

    assert guitar_text.startswith("Bass Guitar (")
    assert bass_text.startswith("Guitar (low) (")
    # Neither part's body may leak into the other.
    assert "Guitar (low) (" not in guitar_text
    assert "Bass Guitar (" not in bass_text
    assert bass_text not in guitar_text
    assert guitar_text not in bass_text
    # The bass has 4 strings, the guitar 6 -- so the row counts differ.
    assert len([l for l in bass_text.splitlines() if "|" in l]) == 4
    assert len([l for l in guitar_text.splitlines() if "|" in l]) == 6
    assert "kick:" in drums_text


def test_ascii_bundle_writes_a_non_empty_bass_tab(tmp_path):
    arrangement = _adversarial_arrangement()
    _write_ascii_bundle(arrangement, str(tmp_path))

    bass = (tmp_path / "bass.tab").read_text()
    guitar = (tmp_path / "guitar.tab").read_text()
    drums = (tmp_path / "drums.txt").read_text()

    assert (tmp_path / "bass.tab").stat().st_size > 0
    assert bass.strip()
    assert guitar.strip()
    assert drums.strip()
    assert bass.strip() not in guitar
    assert "Guitar (low)" in bass
    assert "Guitar (low)" not in guitar
    assert "Bass Guitar" in guitar
    assert "Bass Guitar" not in bass


def test_render_parts_handles_an_empty_arrangement():
    guitar_text, bass_text, drums_text = render_parts(Arrangement(tempo=120.0, ts=(4, 4)))
    assert guitar_text == ""
    assert bass_text == ""
    assert drums_text == "[Drums: no content]"


def _tiny_config():
    return {
        "audio": {"sample_rate": 44100, "decode_channels": 2},
        "features": {
            "n_mels": 32,
            "n_fft": 512,
            "f_min": 30,
            "f_max": 8000,
            "window_frames": 16,
            "hop_ms": 10,
        },
        "model": {"channels": [4, 8], "n_notes": 128},
        "tracking": {},
        "labeling": {"use_llm": False},
        "inference": {"window_overlap": 0.5},
    }


def test_missing_checkpoint_raises_checkpoint_not_found(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(CheckpointNotFound) as exc:
        Converter(_tiny_config(), checkpoint=str(missing))
    assert str(missing) in str(exc.value)
    assert isinstance(exc.value, LyreError)


def test_no_checkpoint_warns_but_constructs(capsys):
    Converter(_tiny_config(), checkpoint=None, device="cpu")
    assert "no checkpoint" in capsys.readouterr().err


def test_config_use_llm_is_ored_with_the_flag(monkeypatch):
    # Turning the LLM on builds the labeler up front, so the environment has to
    # be there; that llm_from_env is called at all is pinned in test_labeling.py.
    monkeypatch.setenv("LYRE_LLM_ENDPOINT", "https://api.example.com/v1")
    monkeypatch.setenv("LYRE_LLM_KEY", "k")

    config = _tiny_config()
    off = Converter(config, device="cpu")
    assert off.use_llm is False
    assert off.llm is None

    by_flag = Converter(config, device="cpu", use_llm=True)
    assert by_flag.use_llm is True
    assert by_flag.llm is not None

    # The flag can only add: a config that enabled it stays enabled.
    config["labeling"]["use_llm"] = True
    assert Converter(config, device="cpu").use_llm is True


def test_window_overlap_comes_from_the_inference_section():
    config = _tiny_config()
    config["inference"]["window_overlap"] = 0.25
    # A stale reader would find these instead.
    config["tracking"]["window_overlap"] = 0.9
    config["features"]["window_overlap"] = 0.75
    assert Converter(config, device="cpu")._overlap() == 0.25


# The exception hierarchy and the re-export identities live in test_errors.py:
# they are a property of lyre.errors, not of the pipeline.


# =====================================================================
# _shift_notes -- pure block-offset arithmetic
# =====================================================================


def _n(pitch, start, end, velocity=100.0):
    return Note(pitch=pitch, start=start, end=end, velocity=velocity)


def test_shift_notes_applies_the_block_offset():
    out = _shift_notes([_n(60, 0.25, 0.75)], offset=2.0, keep_lo=2.0, keep_hi=3.0)
    assert len(out) == 1
    assert out[0].start == pytest.approx(2.25)
    assert out[0].end == pytest.approx(2.75)
    assert out[0].pitch == 60
    assert out[0].velocity == 100.0


def test_shift_notes_keeps_a_note_exactly_on_the_low_edge():
    # Half-open [keep_lo, keep_hi): dropping the note that lands exactly on the
    # seam is how a block boundary silently eats a downbeat.
    out = _shift_notes([_n(60, 0.0, 0.5)], offset=1.0, keep_lo=1.0, keep_hi=2.0)
    assert [n.start for n in out] == [pytest.approx(1.0)]


def test_shift_notes_rejects_a_note_exactly_on_the_high_edge():
    # The next block owns keep_hi. Keeping it here as well duplicates the note
    # at every seam, which reads as a plausible "doubled" transcription.
    out = _shift_notes([_n(60, 1.0, 1.5)], offset=1.0, keep_lo=1.0, keep_hi=2.0)
    assert out == []


def test_shift_notes_clips_end_to_clip_end():
    out = _shift_notes(
        [_n(60, 0.5, 5.0)], offset=1.0, keep_lo=1.0, keep_hi=2.0, clip_end=3.0
    )
    assert out[0].start == pytest.approx(1.5)
    assert out[0].end == pytest.approx(3.0)


def test_shift_notes_never_produces_a_negative_duration():
    # clip_end earlier than the (shifted) onset must collapse to a zero-length
    # note, not to end < start, which every downstream renderer mis-draws.
    out = _shift_notes(
        [_n(60, 0.9, 1.0)], offset=1.0, keep_lo=1.0, keep_hi=2.0, clip_end=1.5
    )
    assert len(out) == 1
    assert out[0].end >= out[0].start


def test_shift_notes_rejects_a_note_before_the_core_window():
    # Pad region on the left: those notes belong to the previous block.
    out = _shift_notes([_n(60, 0.0, 0.1)], offset=0.75, keep_lo=1.0, keep_hi=2.0)
    assert out == []


# =====================================================================
# _chunked_notes -- the block seam
# =====================================================================


CHUNK_SR = 1000
CHUNK_TOTAL_SEC = 3.5
# Deliberately dyadic so block-relative + offset arithmetic is exact, and
# deliberately placed ON the 1.0 / 2.0 / 3.0 block seams.
CHUNK_ONSETS = [
    (0.0, 40),
    (0.25, 45),
    (0.5, 50),
    (1.0, 55),
    (1.75, 57),
    (2.0, 59),
    (2.5, 62),
    (3.0, 64),
    (3.25, 67),
]
CHUNK_NOTE_LEN = 0.0625


def _encoded_waveform():
    """A waveform whose *content* encodes the notes to be found in it.

    The stub transcriber below sees only a block, exactly as the real one does,
    and must derive block-relative times from it. That is what makes the chunked
    result comparable to the unchunked one.
    """
    wav = torch.zeros(1, int(CHUNK_TOTAL_SEC * CHUNK_SR))
    for t, pitch in CHUNK_ONSETS:
        wav[0, int(round(t * CHUNK_SR))] = pitch / 127.0
    return wav


def _stub_block_notes(note_len=CHUNK_NOTE_LEN):
    def _block_notes(self, waveform, sample_rate, no_separate=False, silent=None):
        notes = []
        row = waveform[0]
        for idx in torch.nonzero(row).flatten().tolist():
            start = idx / float(sample_rate)
            pitch = int(round(float(row[idx]) * 127))
            notes.append(_n(pitch, start, start + note_len))
        return {"guitar": notes, "drums": []}

    return _block_notes


def _key(notes):
    return [(n.pitch, round(n.start, 6), round(n.end, 6)) for n in notes]


def _chunk_config(chunk_sec):
    config = _tiny_config()
    config["audio"]["sample_rate"] = CHUNK_SR
    config["inference"]["chunk_sec"] = chunk_sec
    return config


def _transcribed(monkeypatch, chunk_sec, note_len=CHUNK_NOTE_LEN):
    monkeypatch.setattr(Converter, "_block_notes", _stub_block_notes(note_len))
    conv = Converter(_chunk_config(chunk_sec), device="cpu")
    return conv.transcribe(_encoded_waveform(), CHUNK_SR)


def test_chunked_transcription_equals_the_unchunked_reference(monkeypatch):
    reference = _transcribed(monkeypatch, chunk_sec=0)
    chunked = _transcribed(monkeypatch, chunk_sec=1.0)

    assert [i.name for i in chunked] == [i.name for i in reference] == ["drums", "guitar"]
    ref_guitar = next(i for i in reference if i.name == "guitar")
    got_guitar = next(i for i in chunked if i.name == "guitar")

    assert _key(ref_guitar.notes) == [
        (p, round(t, 6), round(t + CHUNK_NOTE_LEN, 6)) for t, p in CHUNK_ONSETS
    ]
    # The whole point: blocking must be invisible in the output.
    assert _key(got_guitar.notes) == _key(ref_guitar.notes)


def test_chunked_transcription_never_duplicates_a_note_at_a_seam(monkeypatch):
    chunked = _transcribed(monkeypatch, chunk_sec=1.0)
    notes = next(i for i in chunked if i.name == "guitar").notes
    pairs = [(n.pitch, round(n.start, 6)) for n in notes]
    assert len(pairs) == len(set(pairs))
    assert len(pairs) == len(CHUNK_ONSETS)


def test_chunked_transcription_output_is_sorted(monkeypatch):
    notes = next(
        i for i in _transcribed(monkeypatch, chunk_sec=1.0) if i.name == "guitar"
    ).notes
    assert notes == sorted(notes, key=lambda n: (n.start, n.pitch))


def test_chunked_transcription_keeps_the_last_block(monkeypatch):
    # The final partial block (3.0 -> 3.5) must not be dropped; its notes are the
    # end of the song and their absence looks like a fade-out.
    notes = next(
        i for i in _transcribed(monkeypatch, chunk_sec=1.0) if i.name == "guitar"
    ).notes
    assert [n.pitch for n in notes if n.start >= 3.0 - 1e-9] == [64, 67]


def test_chunked_transcription_clips_to_the_track_duration(monkeypatch):
    # note_len far longer than the tail block: nothing may end after the track.
    chunked = _transcribed(monkeypatch, chunk_sec=1.0, note_len=2.0)
    notes = next(i for i in chunked if i.name == "guitar").notes
    assert notes
    assert max(n.end for n in notes) <= CHUNK_TOTAL_SEC + 1e-9
    assert all(n.end >= n.start for n in notes)


def test_chunk_sec_larger_than_the_track_takes_the_single_block_path(monkeypatch):
    calls = []
    stub = _stub_block_notes()

    def counting(self, waveform, sample_rate, no_separate=False, silent=None):
        calls.append(waveform.shape[-1])
        return stub(self, waveform, sample_rate, no_separate, silent)

    monkeypatch.setattr(Converter, "_block_notes", counting)
    conv = Converter(_chunk_config(600.0), device="cpu")
    conv.transcribe(_encoded_waveform(), CHUNK_SR)
    assert calls == [int(CHUNK_TOTAL_SEC * CHUNK_SR)]


def test_chunked_path_actually_blocks(monkeypatch):
    calls = []
    stub = _stub_block_notes()

    def counting(self, waveform, sample_rate, no_separate=False, silent=None):
        calls.append(waveform.shape[-1])
        return stub(self, waveform, sample_rate, no_separate, silent)

    monkeypatch.setattr(Converter, "_block_notes", counting)
    conv = Converter(_chunk_config(1.0), device="cpu")
    conv.transcribe(_encoded_waveform(), CHUNK_SR)
    # 4 blocks for 3.5s at 1.0s, and every block is bounded by chunk + 2*pad.
    assert len(calls) == 4
    assert max(calls) <= int((1.0 + 2 * 0.25) * CHUNK_SR)


# =====================================================================
# Separator.separate returns (stems, rate) -- the rate must be USED
# =====================================================================


class _FakeSeparator:
    """Returns a guitar stem, a silent drums stem, and a declared rate."""

    def __init__(self, rate_divisor=1):
        self.rate_divisor = rate_divisor
        self.calls = []

    def separate(self, wav, sr):
        self.calls.append((wav.shape[-1], sr))
        return {"guitar": wav, "drums": torch.zeros_like(wav)}, sr // self.rate_divisor


def _duration_notes(model, waveform, sample_rate, feats, tracking, device, overlap=0.5):
    """Stand-in transcriber whose note timing depends on samples / rate.

    A real transcriber's note times are exactly this quotient, so feeding it the
    caller's rate instead of the separator's rate scales every timestamp -- and
    the note count, pitch range and shapes all stay plausible.
    """
    return [_n(60, 0.0, waveform.shape[-1] / float(sample_rate))]


def _separator_converter(monkeypatch, rate_divisor=1, sr=8000):
    monkeypatch.setattr(pipeline_mod, "_notes_from_stem", _duration_notes)
    config = _tiny_config()
    config["audio"]["sample_rate"] = sr
    config["inference"]["chunk_sec"] = 0
    conv = Converter(config, device="cpu")
    conv.separator = _FakeSeparator(rate_divisor=rate_divisor)
    return conv


def test_transcribe_keeps_a_silent_stem_as_an_empty_instrument(monkeypatch):
    conv = _separator_converter(monkeypatch)
    instruments = conv.transcribe(torch.ones(1, 8000), 8000)
    assert [i.name for i in instruments] == ["drums", "guitar"]
    drums = next(i for i in instruments if i.name == "drums")
    assert drums.notes == []
    assert drums.is_drum is True


def test_transcribe_records_an_advisory_note_for_a_silent_stem(monkeypatch, capsys):
    conv = _separator_converter(monkeypatch)
    notes = []
    conv.transcribe(torch.ones(1, 8000), 8000, notes=notes)
    assert len(notes) == 1
    assert "drums" in notes[0]
    assert "silent" in notes[0]
    # It is an advisory, not a failure: it is streamed to stderr, never stdout.
    captured = capsys.readouterr()
    assert "drums" in captured.err
    assert captured.out == ""


class _SometimesSilentSeparator:
    """Silent drums in the first block only."""

    def __init__(self):
        self.blocks = 0

    def separate(self, wav, sr):
        self.blocks += 1
        drums = torch.zeros_like(wav) if self.blocks == 1 else torch.ones_like(wav)
        return {"guitar": wav, "drums": drums}, sr


def test_a_stem_silent_in_only_one_block_is_not_reported(monkeypatch):
    # A stem the separator found nothing in for one block of a long track is not
    # an empty instrument; saying so is both repetitive and untrue.
    monkeypatch.setattr(pipeline_mod, "_notes_from_stem", _duration_notes)
    config = _tiny_config()
    config["audio"]["sample_rate"] = 8000
    config["inference"]["chunk_sec"] = 1.0
    config["inference"]["chunk_pad_sec"] = 0.0
    conv = Converter(config, device="cpu")
    conv.separator = _SometimesSilentSeparator()

    notes = []
    conv.transcribe(torch.ones(1, 24000), 8000, notes=notes)

    assert conv.separator.blocks == 3
    assert notes == []


def test_a_stem_silent_in_every_block_is_reported_once(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "_notes_from_stem", _duration_notes)
    config = _tiny_config()
    config["audio"]["sample_rate"] = 8000
    config["inference"]["chunk_sec"] = 1.0
    config["inference"]["chunk_pad_sec"] = 0.0
    conv = Converter(config, device="cpu")
    conv.separator = _FakeSeparator()

    notes = []
    conv.transcribe(torch.ones(1, 24000), 8000, notes=notes)

    assert len(conv.separator.calls) == 3
    # Three blocks, one advisory.
    assert len(notes) == 1
    assert "drums" in notes[0]


def test_transcribe_uses_the_rate_the_separator_returned(monkeypatch):
    # 1 second of audio at the caller's rate. The separator declares half that
    # rate, so the stem is 2 seconds long and the note must be 2 seconds long.
    same = _separator_converter(monkeypatch, rate_divisor=1)
    halved = _separator_converter(monkeypatch, rate_divisor=2)

    at_full = next(
        i for i in same.transcribe(torch.ones(1, 8000), 8000) if i.name == "guitar"
    ).notes
    at_half = next(
        i for i in halved.transcribe(torch.ones(1, 8000), 8000) if i.name == "guitar"
    ).notes

    assert at_full[0].end == pytest.approx(1.0)
    assert at_half[0].end == pytest.approx(2.0)


def test_no_separate_path_uses_the_caller_rate_and_skips_the_separator(monkeypatch):
    conv = _separator_converter(monkeypatch)
    instruments = conv.transcribe(torch.ones(1, 8000), 8000, no_separate=True)
    assert [i.name for i in instruments] == ["other"]
    assert instruments[0].notes[0].end == pytest.approx(1.0)
    assert conv.separator.calls == []


# =====================================================================
# _stage -- isolation and cleanup
# =====================================================================


def _converter(monkeypatch=None):
    return Converter(_tiny_config(), device="cpu")


def test_stage_success_returns_paths_and_records_nothing(tmp_path):
    conv = _converter()
    path = str(tmp_path / "out.txt")
    failures = []
    written = conv._stage("thing", lambda: open(path, "w").close(), failures, paths=(path,))
    assert written == [path]
    assert failures == []


def test_stage_only_returns_paths_that_actually_exist(tmp_path):
    # A stage that claims two outputs but writes one must not report the other --
    # and must say so, because an external tool exiting 0 while writing nothing
    # otherwise leaves the user with no error and no file.
    conv = _converter()
    made = str(tmp_path / "made.txt")
    never = str(tmp_path / "never.txt")
    failures = []
    written = conv._stage(
        "thing", lambda: open(made, "w").close(), failures, paths=(made, never)
    )
    assert written == [made]
    assert len(failures) == 1
    assert "thing" in failures[0]
    assert "never.txt" in failures[0]
    assert "did not write" in failures[0]


def test_stage_failure_records_one_message_and_deletes_what_it_wrote(tmp_path):
    conv = _converter()
    half = tmp_path / "half.txt"
    other = tmp_path / "untouched.txt"
    other.write_text("belongs to another stage")

    failures = []

    def boom():
        half.write_text("truncated garbage from the failed writer")
        raise ValueError("disk went away")

    written = conv._stage("musicxml", boom, failures, paths=(str(half), str(other)))

    assert written == []
    assert len(failures) == 1
    assert "musicxml" in failures[0]
    assert "disk went away" in failures[0]
    # What this run wrote is removed; what it never touched survives untouched.
    assert not half.exists()
    assert other.read_text() == "belongs to another stage"


def test_stage_failure_keeps_a_pre_existing_output_it_never_touched(tmp_path):
    # The previous run's good output is the whole point: a stage that fails
    # before writing anything must not take it down with it.
    conv = _converter()
    previous = tmp_path / "score.musicxml"
    previous.write_text("<score>LAST RUN</score>")
    failures = []

    def boom():
        raise ValueError("nothing was written")

    written = conv._stage("musicxml", boom, failures, paths=(str(previous),))

    assert written == []
    assert previous.read_text() == "<score>LAST RUN</score>"


def test_stage_failure_removes_a_pre_existing_output_it_replaced(tmp_path):
    # MuseScore writes its PDF in place, so a run that replaced last run's file
    # and then failed leaves a file whose contents belong to neither run.
    conv = _converter()
    target = tmp_path / "score.pdf"
    target.write_text("last run's pdf")
    failures = []

    def boom():
        target.write_text("this run's half-written pdf")
        raise ValueError("mscore died")

    conv._stage("pdf", boom, failures, paths=(str(target),))

    assert not target.exists()


def test_stage_routes_an_advisory_exception_to_notes(tmp_path):
    from lyre.errors import PdfBackendMissing

    conv = _converter()
    failures = []
    notes = []
    path = str(tmp_path / "score.pdf")

    def boom():
        raise PdfBackendMissing("no MuseScore CLI on PATH")

    written = conv._stage(
        "pdf",
        boom,
        failures,
        paths=(path,),
        notes=notes,
        advisory=(PdfBackendMissing,),
    )

    assert written == []
    assert failures == []
    assert len(notes) == 1
    assert "no MuseScore CLI on PATH" in notes[0]


def test_stage_keeps_a_render_failure_on_the_failure_channel(tmp_path):
    # PdfUnavailable is not in `advisory`: MuseScore ran and did not produce the
    # PDF the user asked for, which is a failure of the run.
    from lyre.errors import PdfBackendMissing

    conv = _converter()
    failures = []
    notes = []
    path = str(tmp_path / "score.pdf")

    def boom():
        raise PdfUnavailable("mscore failed with exit code 1: boom")

    conv._stage(
        "pdf", boom, failures, paths=(path,), notes=notes,
        advisory=(PdfBackendMissing,),
    )

    assert notes == []
    assert len(failures) == 1
    assert "boom" in failures[0]


# =====================================================================
# _atomic_write -- no partial file may survive
# =====================================================================


def test_atomic_write_renames_the_scratch_file_into_place(tmp_path):
    target = tmp_path / "x.musicxml"
    pipeline_mod._atomic_write(
        str(target), lambda tmp: open(tmp, "w").write("<score/>")
    )
    assert target.read_text() == "<score/>"
    assert os.listdir(tmp_path) == ["x.musicxml"]


def test_a_failed_writer_leaves_no_tmp_file(tmp_path):
    # The scratch file is only consumed by os.replace on the success path, so a
    # writer that dies part-way through is the only thing that can strand one --
    # and a stray .tmp in the output directory is what every later existence
    # check mistakes for a real output.
    def writer(tmp):
        open(tmp, "w").write("half a score")
        raise RuntimeError("disk went away")

    with pytest.raises(RuntimeError):
        pipeline_mod._atomic_write(str(tmp_path / "x.musicxml"), writer)
    assert os.listdir(tmp_path) == []


def test_atomic_write_does_not_use_a_guessable_scratch_name(tmp_path):
    # A predictable "<path>.tmp" is guessable by another user in a shared output
    # directory, and two concurrent runs writing one score would interleave.
    seen = []
    target = tmp_path / "score.musicxml"

    def writer(tmp):
        seen.append(os.path.basename(tmp))
        open(tmp, "w").write("<score/>")

    pipeline_mod._atomic_write(str(target), writer)

    assert seen[0] != "score.musicxml.tmp"
    assert seen[0].startswith(".lyre-")
    assert seen[0].endswith(".tmp")


# =====================================================================
# convert -- the bundle contract
# =====================================================================


def _convert_config(sr=8000):
    config = _tiny_config()
    config["audio"]["sample_rate"] = sr
    config["inference"]["chunk_sec"] = 0
    config["arrange"] = {"tempo": 120.0, "time_signature": [4, 4]}
    return config


def _pipeline_notes():
    return {
        "guitar": _notes([64, 59, 55, 52, 57, 62, 64, 59]),
        "bass": _notes([28, 33, 38, 43, 30, 35, 40, 45]),
        "drums": [
            Note(pitch=36, start=0.0, end=0.1, velocity=100.0),
            Note(pitch=38, start=0.5, end=0.6, velocity=100.0),
            Note(pitch=42, start=1.0, end=1.1, velocity=100.0),
        ],
    }


def _checkpoint(tmp_path, config):
    """A real state dict for the tiny model, so no run is "untrained".

    Without one every ``convert`` reports a failure -- deliberately, because
    untrained weights produce tabs that look plausible and mean nothing.
    """
    from lyre.transcriber.model import MultiPitchNet

    model = MultiPitchNet(
        n_mels=config["features"]["n_mels"],
        n_notes=config["model"]["n_notes"],
        channels=config["model"]["channels"],
    )
    path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), str(path))
    return str(path)


def _pipeline_setup(monkeypatch, tmp_path, sr=8000):
    """A Converter wired to stubs: no model inference, no MuseScore, real files."""
    monkeypatch.setattr(
        Converter,
        "_block_notes",
        lambda self, wav, rate, no_separate=False, silent=None: _pipeline_notes(),
    )
    pdf_calls = []

    def fake_pdf(xml_path, pdf_path):
        pdf_calls.append((xml_path, pdf_path))
        with open(pdf_path, "wb") as fh:
            fh.write(b"%PDF-1.4\n")

    monkeypatch.setattr(pipeline_mod, "write_pdf", fake_pdf)

    audio = str(tmp_path / "in.wav")
    save_wav(audio, torch.zeros(1, sr), sr)
    out_dir = str(tmp_path / "out")
    config = _convert_config(sr)
    conv = Converter(config, device="cpu", checkpoint=_checkpoint(tmp_path, config))
    return conv, audio, out_dir, pdf_calls


def test_convert_writes_the_whole_bundle(monkeypatch, tmp_path):
    conv, audio, out_dir, pdf_calls = _pipeline_setup(monkeypatch, tmp_path)
    result = conv.convert(audio, out_dir)

    assert result["failures"] == []
    files = set(result["files"])
    for expected in (
        "score.mid",
        "score.musicxml",
        "score.gp5",
        "score.pdf",
        "guitar.tab",
        "bass.tab",
        "drums.txt",
    ):
        assert expected in files, f"{expected} missing from {sorted(files)}"
    # `files` reports what this run wrote, so every name must exist on disk.
    for name in files:
        assert os.path.exists(os.path.join(out_dir, name))
    assert len(pdf_calls) == 1


def test_convert_leaves_no_tmp_files_behind(monkeypatch, tmp_path):
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    conv.convert(audio, out_dir)
    assert [f for f in os.listdir(out_dir) if f.endswith(".tmp")] == []


def test_convert_bass_tab_is_non_empty_and_disjoint_from_guitar(monkeypatch, tmp_path):
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    conv.convert(audio, out_dir)

    bass = open(os.path.join(out_dir, "bass.tab")).read()
    guitar = open(os.path.join(out_dir, "guitar.tab")).read()
    assert bass.strip()
    assert guitar.strip()
    assert os.path.getsize(os.path.join(out_dir, "bass.tab")) > 0
    assert bass.strip() not in guitar
    assert guitar.strip() not in bass
    # A bass part has 4 string rows; the guitar 6. Writing the guitar rendering
    # into bass.tab is the failure this pins.
    assert len([l for l in bass.splitlines() if "|" in l]) == 4
    assert len([l for l in guitar.splitlines() if "|" in l]) == 6


def test_convert_reports_advisory_notes_separately_from_failures(monkeypatch, tmp_path):
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)

    def noisy(self, wav, rate, no_separate=False, silent=None):
        silent.append({"vocals"})
        return _pipeline_notes()

    monkeypatch.setattr(Converter, "_block_notes", noisy)
    result = conv.convert(audio, out_dir)

    assert result["failures"] == []
    assert any("vocals" in n for n in result["notes"])
    # Regression: notes and failures shared one list, so an advisory made a
    # perfectly good conversion exit 1.
    assert result["notes"] is not result["failures"]


def test_convert_streams_each_message_exactly_once(monkeypatch, tmp_path, capsys):
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)

    def noisy(self, wav, rate, no_separate=False, silent=None):
        silent.append({"vocals"})
        return _pipeline_notes()

    monkeypatch.setattr(Converter, "_block_notes", noisy)
    result = conv.convert(audio, out_dir)

    message = next(n for n in result["notes"] if "vocals" in n)
    err = capsys.readouterr().err
    assert err.count(message) == 1


def test_convert_survives_one_failing_renderer(monkeypatch, tmp_path):
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline_mod, "write_gp5", _boom("gp5 writer exploded")
    )
    result = conv.convert(audio, out_dir)

    files = set(result["files"])
    assert "score.gp5" not in files
    assert not os.path.exists(os.path.join(out_dir, "score.gp5"))
    # Everything else still ships.
    for expected in ("score.mid", "guitar.tab", "score.musicxml", "score.pdf"):
        assert expected in files
    gp5_failures = [f for f in result["failures"] if f.startswith("gp5:")]
    assert len(gp5_failures) == 1
    assert "gp5 writer exploded" in gp5_failures[0]
    assert len(result["failures"]) == 1


def _boom(message):
    def fn(*args, **kwargs):
        raise RuntimeError(message)

    return fn


def test_convert_skips_pdf_but_keeps_a_previous_musicxml(monkeypatch, tmp_path):
    conv, audio, out_dir, pdf_calls = _pipeline_setup(monkeypatch, tmp_path)
    os.makedirs(out_dir, exist_ok=True)
    previous = os.path.join(out_dir, "score.musicxml")
    with open(previous, "w") as fh:
        fh.write("<score>LAST RUN</score>")

    monkeypatch.setattr(pipeline_mod, "write_musicxml", _boom("musicxml exploded"))
    result = conv.convert(audio, out_dir)

    # The failing stage never got as far as replacing the file, so the earlier
    # run's score survives intact...
    assert open(previous).read() == "<score>LAST RUN</score>"
    # ...but it is not this run's output, so it is not reported as one, and the
    # PDF stage must not engrave it. Gating the PDF on os.path.exists() instead
    # would have reported both "musicxml failed" and "score.pdf" -- the PDF
    # being of a different arrangement entirely.
    assert pdf_calls == []
    assert "score.musicxml" not in result["files"]
    assert "score.pdf" not in result["files"]
    assert any(f.startswith("musicxml:") for f in result["failures"])
    assert any("pdf" in f and "skipped" in f for f in result["failures"])


def test_a_failed_second_run_does_not_destroy_the_first_runs_score(
    monkeypatch, tmp_path
):
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)

    first = conv.convert(audio, out_dir)
    assert first["failures"] == []
    score = os.path.join(out_dir, "score.musicxml")
    good = open(score).read()
    assert good.strip()

    monkeypatch.setattr(pipeline_mod, "write_musicxml", _boom("musicxml exploded"))
    second = conv.convert(audio, out_dir)

    # Both directions of the same invariant: the good file is still on disk with
    # its original bytes, and the run that did not write it does not claim it.
    assert open(score).read() == good
    assert "score.musicxml" not in second["files"]
    assert any(f.startswith("musicxml:") for f in second["failures"])


def test_convert_files_are_not_a_directory_listing(monkeypatch, tmp_path):
    # A junk file left in out_dir by something else must never be reported as an
    # output of this run.
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "leftover.gp5"), "w") as fh:
        fh.write("not ours")

    result = conv.convert(audio, out_dir)
    assert "leftover.gp5" not in result["files"]


def test_convert_reports_instrument_names(monkeypatch, tmp_path):
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    result = conv.convert(audio, out_dir)
    assert result["instruments"]
    assert "bass" in result["instruments"]
    assert any(n.startswith("guitar") for n in result["instruments"])
    assert "drums" in result["instruments"]


def test_convert_propagates_a_decode_failure(monkeypatch, tmp_path):
    conv, _, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    junk = tmp_path / "junk.wav"
    junk.write_text("not audio")
    # Decode is deliberately fatal: nothing downstream means anything.
    with pytest.raises(AudioDecodeError):
        conv.convert(str(junk), out_dir)


def test_convert_without_a_checkpoint_reports_a_failure(monkeypatch, tmp_path):
    # The warning has no channel to go to when it is raised -- __init__ has no
    # result dict -- so it is stashed and replayed onto every run's failures.
    # Untrained weights produce tabs that look plausible and mean nothing, and a
    # `lyre convert && publish` caller must not treat them as a result.
    _, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    untrained = Converter(_convert_config(8000), device="cpu")

    result = untrained.convert(audio, out_dir)

    assert any("no checkpoint" in f for f in result["failures"])
    # It is still a full bundle: the outputs exist, they just cannot be trusted.
    assert "score.mid" in result["files"]


def test_the_startup_failure_is_replayed_on_every_run(monkeypatch, tmp_path):
    _, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    untrained = Converter(_convert_config(8000), device="cpu")

    first = untrained.convert(audio, out_dir)
    second = untrained.convert(audio, out_dir)

    # Recorded, not consumed: a second conversion is no more trustworthy.
    assert len([f for f in first["failures"] if "no checkpoint" in f]) == 1
    assert len([f for f in second["failures"] if "no checkpoint" in f]) == 1


def test_a_missing_pdf_backend_is_a_note_not_a_failure(monkeypatch, tmp_path):
    from lyre.errors import PdfBackendMissing

    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline_mod, "write_pdf", _boom_with(PdfBackendMissing, "no MuseScore on PATH")
    )
    result = conv.convert(audio, out_dir)

    # Every other output was produced correctly; a machine without the optional
    # engraver has not had a failed run.
    assert result["failures"] == []
    assert any("MuseScore" in n for n in result["notes"])
    assert "score.pdf" not in result["files"]
    assert "score.musicxml" in result["files"]


def test_a_crashed_engraver_is_a_failure(monkeypatch, tmp_path):
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline_mod,
        "write_pdf",
        _boom_with(PdfUnavailable, "mscore failed with exit code 1: boom"),
    )
    result = conv.convert(audio, out_dir)

    # MuseScore ran and did not produce the PDF that was asked for.
    assert any("boom" in f for f in result["failures"])
    assert not any("boom" in n for n in result["notes"])


def _boom_with(cls, message):
    def fn(*args, **kwargs):
        raise cls(message)

    return fn


def test_a_stage_that_writes_nothing_is_reported(monkeypatch, tmp_path):
    # An external tool that exits 0 and writes nothing must not be reported as a
    # success: that leaves the user with no error and no file.
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline_mod, "write_pdf", lambda xml, pdf: None)
    result = conv.convert(audio, out_dir)

    assert "score.pdf" not in result["files"]
    assert any("did not write" in f and "score.pdf" in f for f in result["failures"])


def test_the_ascii_bundle_names_come_from_one_constant(monkeypatch, tmp_path):
    # The stage's declared outputs and the writer read the same tuple: a
    # hand-maintained second copy drifts, and the drift is invisible until a
    # failure cleanup misses a file.
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    result = conv.convert(audio, out_dir)
    assert set(pipeline_mod.ASCII_BUNDLE) <= set(result["files"])


def test_a_failing_ascii_stage_removes_the_files_it_already_replaced(
    monkeypatch, tmp_path
):
    conv, audio, out_dir, _ = _pipeline_setup(monkeypatch, tmp_path)
    conv.convert(audio, out_dir)
    before = {
        name: open(os.path.join(out_dir, name)).read()
        for name in pipeline_mod.ASCII_BUNDLE
    }

    real_render = pipeline_mod.render_parts

    def half_render(arrangement):
        guitar, bass, drums = real_render(arrangement)
        return [guitar, _Exploding(), drums]

    monkeypatch.setattr(pipeline_mod, "render_parts", half_render)
    result = conv.convert(audio, out_dir)

    # guitar.tab was already replaced when bass.tab blew up, so it holds neither
    # run's content and must go; nothing may claim to have been written.
    assert not os.path.exists(os.path.join(out_dir, "guitar.tab"))
    assert not any(f in result["files"] for f in pipeline_mod.ASCII_BUNDLE)
    assert any(f.startswith("ascii tabs:") for f in result["failures"])
    assert before  # the first run really did write them


class _Exploding:
    """Stands in for a rendered part that fails at write time."""

    def __add__(self, other):
        raise RuntimeError("renderer exploded")

    def __bool__(self):
        raise RuntimeError("renderer exploded")
