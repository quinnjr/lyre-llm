# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

"""CLI surface: flag defaults, and the exit-code contract.

The exit code is the only thing a script calling ``lyre`` can see. Getting it
wrong is silent by construction: a run that produced every output but exited 1
breaks every ``set -e`` caller, and a run that lost an output but exited 0 is
never noticed at all.
"""

import pytest

import importlib

from lyre.cli.main import _exit_code, build_parser, cmd_convert, main

# `lyre.cli.__init__` re-exports the *function* `main`, so `import lyre.cli.main`
# binds the function, not the module. Fetch the module explicitly.
cli = importlib.import_module("lyre.cli.main")
from lyre.errors import ConfigError, LabelingError, LyreError


# ------------------------------------------------------------------ build_parser


def _parse(argv):
    return build_parser().parse_args(argv)


def test_convert_flag_defaults():
    args = _parse(["convert", "song.mp3"])
    assert args.audio == "song.mp3"
    assert args.out == "out"
    assert args.config == "configs/default.yaml"
    assert args.checkpoint is None
    assert args.model == "htdemucs_6s"
    assert args.device == "auto"
    assert args.name == "score"
    assert args.llm is False
    assert args.no_separate is False
    assert args.func is cmd_convert


def test_convert_flags_are_settable():
    args = _parse(
        [
            "convert", "song.mp3",
            "-o", "outdir",
            "--checkpoint", "best.pt",
            "--model", "htdemucs",
            "--device", "cpu",
            "--name", "tune",
            "--llm",
            "--no-separate",
        ]
    )
    assert args.out == "outdir"
    assert args.checkpoint == "best.pt"
    assert args.model == "htdemucs"
    assert args.device == "cpu"
    assert args.name == "tune"
    assert args.llm is True
    assert args.no_separate is True


def test_export_int8_maps_to_quantize_and_defaults_off():
    off = _parse(["export", "--checkpoint", "best.pt"])
    assert off.quantize is False
    assert off.out == "model.onnx"
    # The flag is spelled --int8 but the handler reads args.quantize; a rename on
    # one side only would leave quantization permanently off.
    assert not hasattr(off, "int8")

    on = _parse(["export", "--checkpoint", "best.pt", "--int8"])
    assert on.quantize is True


def test_eval_separator_flags():
    args = _parse(["eval", "--checkpoint", "best.pt"])
    assert args.through_separator is False
    assert args.separator_model == "htdemucs_6s"
    assert args.index is None
    assert args.device == "auto"

    args = _parse(
        ["eval", "--checkpoint", "best.pt", "--through-separator",
         "--separator-model", "htdemucs"]
    )
    assert args.through_separator is True
    assert args.separator_model == "htdemucs"


def test_eval_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        _parse(["eval"])


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        _parse([])


def test_prepare_data_and_train_defaults():
    assert _parse(["prepare-data"]).config == "configs/default.yaml"
    train = _parse(["train"])
    assert train.devices == "auto"
    assert train.epochs is None
    assert train.resume is None


# --------------------------------------------------------------------- _exit_code


@pytest.mark.parametrize(
    "result,expected",
    [
        (None, 0),
        ({}, 0),
        ({"failures": [], "notes": ["silent stem"]}, 0),
        ({"failures": ["gp5: boom"]}, 1),
        ("not a dict", 0),
    ],
)
def test_exit_code_mapping(result, expected):
    assert _exit_code(result) == expected


def test_exit_code_ignores_a_loosely_named_warnings_key():
    # Only `failures` is "an output you asked for is missing". Honouring a second
    # list would bring back "any advisory means exit 1", which is what made every
    # successful conversion look like a failed one.
    assert _exit_code({"warnings": ["something"]}) == 0
    assert _exit_code({"failures": [], "warnings": ["something"]}) == 0


# ------------------------------------------------------------------------- main


def _run(monkeypatch, handler, argv=("convert", "song.mp3")):
    monkeypatch.setattr(cli, "cmd_convert", handler)
    return main(list(argv))


def test_main_returns_the_handlers_code(monkeypatch):
    assert _run(monkeypatch, lambda args: 0) == 0
    assert _run(monkeypatch, lambda args: 1) == 1


def test_main_treats_a_none_return_as_success(monkeypatch):
    assert _run(monkeypatch, lambda args: None) == 0


def test_main_maps_lyre_error_to_2(monkeypatch, capsys):
    def boom(args):
        raise LabelingError("LYRE_LLM_ENDPOINT is not set")

    assert _run(monkeypatch, boom) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "LYRE_LLM_ENDPOINT is not set" in err
    assert "Traceback" not in err


def test_main_maps_file_not_found_to_2(monkeypatch, capsys):
    def boom(args):
        raise FileNotFoundError("no such file: song.mp3")

    assert _run(monkeypatch, boom) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_main_maps_keyboard_interrupt_to_130(monkeypatch, capsys):
    def boom(args):
        raise KeyboardInterrupt

    assert _run(monkeypatch, boom) == 130
    assert "interrupted" in capsys.readouterr().err


def test_main_does_not_swallow_a_genuine_bug(monkeypatch):
    # Only LyreError / FileNotFoundError / KeyboardInterrupt are translated. A
    # TypeError from a real bug must not be reported as "error: ..." with exit 2.
    def boom(args):
        raise TypeError("unsupported operand")

    with pytest.raises(TypeError):
        _run(monkeypatch, boom)


def test_a_malformed_config_exits_2_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "broken.yaml"
    bad.write_text("audio: {sample_rate: 44100\nfeatures: [\n")

    code = main(["convert", "song.mp3", "--config", str(bad)])

    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert str(bad) in err
    assert "Traceback" not in err
    assert issubclass(ConfigError, LyreError)


def test_a_missing_config_exits_2(tmp_path, capsys):
    missing = tmp_path / "nope.yaml"
    code = main(["convert", "song.mp3", "--config", str(missing)])
    assert code == 2
    assert capsys.readouterr().err.startswith("error: ")


# ------------------------------------------------------- cmd_convert exit codes


class _FakeConverter:
    """Stands in for Converter: records construction args, returns a fixed dict."""

    last = None

    def __init__(self, config, checkpoint=None, device="auto", use_llm=False, model_name=None):
        self.kwargs = dict(
            checkpoint=checkpoint, device=device, use_llm=use_llm, model_name=model_name
        )
        _FakeConverter.last = self
        self.result = None

    def convert(self, audio, out, name="score", no_separate=False):
        self.call = dict(audio=audio, out=out, name=name, no_separate=no_separate)
        return self.result


def _fake_convert(monkeypatch, result, stream=()):
    monkeypatch.setattr(cli, "load_config", lambda path: {})

    def factory(*args, **kwargs):
        conv = _FakeConverter(*args, **kwargs)
        conv.result = result

        def convert(audio, out, name="score", no_separate=False):
            from lyre.reporting import warn

            for message in stream:
                warn(message)
            return result

        conv.convert = convert
        return conv

    monkeypatch.setattr(cli, "Converter", factory)


def test_convert_exits_0_when_only_advisory_notes_were_recorded(monkeypatch, capsys):
    # THE regression: notes are routine on real input (a silent stem, an octave
    # shift the arranger had to make). Exiting 1 for them made every successful
    # conversion look like a failure.
    _fake_convert(
        monkeypatch,
        {
            "out_dir": "out",
            "instruments": ["bass", "drums", "guitar"],
            "files": ["score.mid", "score.gp5"],
            "failures": [],
            "notes": ["stem 'piano': nothing detected (silent)", "guitar: shifted an octave"],
        },
    )
    assert main(["convert", "song.mp3"]) == 0
    out = capsys.readouterr()
    assert "instruments: bass, drums, guitar" in out.out
    assert "score.mid" in out.out
    assert "2 note(s)" in out.err


def test_convert_exits_1_when_a_stage_failed(monkeypatch, capsys):
    _fake_convert(
        monkeypatch,
        {
            "out_dir": "out",
            "instruments": ["guitar"],
            "files": ["score.mid"],
            "failures": ["gp5: writer exploded"],
            "notes": [],
        },
    )
    assert main(["convert", "song.mp3"]) == 1
    assert "1 stage failure(s)" in capsys.readouterr().err


def test_convert_exits_0_for_a_clean_run(monkeypatch):
    _fake_convert(
        monkeypatch,
        {"out_dir": "out", "instruments": ["guitar"], "files": ["score.mid"],
         "failures": [], "notes": []},
    )
    assert main(["convert", "song.mp3"]) == 0


def test_convert_summarises_rather_than_reprinting(monkeypatch, capsys):
    message = "stem 'piano': nothing detected (silent); merging as an empty track"
    _fake_convert(
        monkeypatch,
        {"out_dir": "out", "instruments": ["guitar"], "files": ["score.mid"],
         "failures": [], "notes": [message]},
        stream=(message,),
    )
    assert main(["convert", "song.mp3"]) == 0
    err = capsys.readouterr().err
    # warn() already streamed it; the CLI must only add a count.
    assert err.count(message) == 1
    assert "1 note(s)" in err


def test_convert_passes_its_flags_through(monkeypatch):
    _fake_convert(
        monkeypatch,
        {"instruments": [], "files": [], "failures": [], "notes": []},
    )
    main(
        ["convert", "song.mp3", "-o", "outdir", "--checkpoint", "best.pt",
         "--device", "cpu", "--llm", "--model", "htdemucs", "--name", "tune"]
    )
    assert _FakeConverter.last.kwargs == dict(
        checkpoint="best.pt", device="cpu", use_llm=True, model_name="htdemucs"
    )


def test_convert_writes_only_the_listing_to_stdout(monkeypatch, capsys):
    # Advisories on stdout would corrupt anything parsing the file listing.
    _fake_convert(
        monkeypatch,
        {"instruments": ["guitar"], "files": ["score.mid"],
         "failures": ["gp5: boom"], "notes": ["a note"]},
    )
    main(["convert", "song.mp3"])
    out = capsys.readouterr().out
    assert "gp5: boom" not in out
    assert "a note" not in out


# ------------------------------------------------------------------ --name
#
# `--name` is interpolated into every output path, and those paths are deleted
# when a stage fails, so a separator or `..` in it lets the run write and unlink
# outside `--out`. It is rejected before anything else happens.


def _refuse_everything(monkeypatch):
    """Make any work past name validation an immediate, loud test failure."""

    def boom(*args, **kwargs):
        raise AssertionError("work started despite an invalid --name")

    monkeypatch.setattr(cli, "load_config", boom)
    monkeypatch.setattr(cli, "Converter", boom)


@pytest.mark.parametrize(
    "name", ["../../x", "sub/score", "..", "", "   ", "sc\0ore"]
)
def test_an_unsafe_name_exits_2_before_any_work(monkeypatch, capsys, name):
    _refuse_everything(monkeypatch)
    assert main(["convert", "song.mp3", "--name", name]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "--name" in err
    assert "Traceback" not in err


def test_a_plain_name_is_accepted(monkeypatch):
    _fake_convert(
        monkeypatch,
        {"instruments": [], "files": [], "failures": [], "notes": []},
    )
    assert main(["convert", "song.mp3", "--name", "my tune.v2"]) == 0
    assert _FakeConverter.last is not None


# -------------------------------------------------- converter startup config
#
# Every inference knob is bounded in `Converter.__init__`, before separation and
# transcription burn minutes of GPU time on a run that cannot finish.


@pytest.mark.parametrize(
    "section",
    [
        {"chunk_sec": -1.0},
        {"chunk_pad_sec": "soon"},
        # 1.0 advances the analysis window one frame at a time: not an error at
        # runtime, just a run that never ends.
        {"window_overlap": 1.0},
        {"window_overlap": -0.5},
    ],
)
def test_a_bad_inference_knob_exits_2_from_the_constructor(
    monkeypatch, capsys, tmp_path, section
):
    config = dict(_pipeline_config())
    config["inference"] = dict(config["inference"], **section)
    monkeypatch.setattr(cli, "load_config", lambda path: config)

    assert main(["convert", str(_silent_wav(tmp_path)), "-o", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err


# ------------------------------------------------- end-to-end exit codes
#
# The exit code is all a `set -e` caller can see, so it is pinned against the
# real Converter rather than against a stand-in that returns a fixed dict.


def _pipeline_config():
    return {
        "audio": {"sample_rate": 8000, "decode_channels": 2},
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
        "inference": {"window_overlap": 0.5, "chunk_sec": 0},
        "arrange": {"tempo": 120.0, "time_signature": [4, 4]},
    }


def _stub_notes():
    from lyre.tracking.hmm import Note

    def line(pitches):
        return [
            Note(pitch=p, start=i * 0.25, end=(i + 1) * 0.25, velocity=100.0)
            for i, p in enumerate(pitches)
        ]

    return {
        "guitar": line([64, 59, 55, 52, 57, 62, 64, 59]),
        "bass": line([28, 33, 38, 43, 30, 35, 40, 45]),
        "drums": line([36, 38, 42]),
    }


def _silent_wav(tmp_path):
    import torch

    from lyre.io.decode import save_wav

    path = tmp_path / "in.wav"
    save_wav(str(path), torch.zeros(1, 8000), 8000)
    return path


def _checkpoint(tmp_path, config):
    import torch

    from lyre.transcriber.model import MultiPitchNet

    model = MultiPitchNet(
        n_mels=config["features"]["n_mels"],
        n_notes=config["model"]["n_notes"],
        channels=config["model"]["channels"],
    )
    path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), str(path))
    return path


def _pipeline_converter_factory(monkeypatch):
    """The real Converter, with model inference stubbed out."""
    from lyre.pipeline import Converter

    monkeypatch.setattr(
        Converter,
        "_block_notes",
        lambda self, wav, rate, no_separate=False, silent=None: _stub_notes(),
    )
    return Converter


def _cli_pipeline(monkeypatch, tmp_path, pdf=None, config=None):
    """Returns the argv prefix for a `convert` run against the real pipeline."""
    from lyre import pipeline as pipeline_mod

    config = config or _pipeline_config()
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "Converter", _pipeline_converter_factory(monkeypatch))

    if pdf is None:
        def pdf(xml_path, pdf_path):
            with open(pdf_path, "wb") as fh:
                fh.write(b"%PDF-1.4\n")

    monkeypatch.setattr(pipeline_mod, "write_pdf", pdf)
    out_dir = tmp_path / "out"
    return ["convert", str(_silent_wav(tmp_path)), "-o", str(out_dir)], out_dir


def test_convert_without_a_checkpoint_exits_1(monkeypatch, tmp_path, capsys):
    # Untrained weights produce tabs that look plausible and mean nothing. That
    # is a failure of the run, not an advisory: `lyre convert && publish` must
    # not treat the output as a result.
    argv, out_dir = _cli_pipeline(monkeypatch, tmp_path)
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "no checkpoint" in err
    assert "stage failure(s)" in err


def test_convert_with_a_checkpoint_and_no_musescore_exits_0(
    monkeypatch, tmp_path, capsys
):
    from lyre.errors import PdfBackendMissing

    def no_binary(xml_path, pdf_path):
        raise PdfBackendMissing("PDF export needs a MuseScore CLI on PATH")

    config = _pipeline_config()
    argv, out_dir = _cli_pipeline(monkeypatch, tmp_path, pdf=no_binary, config=config)
    argv += ["--checkpoint", str(_checkpoint(tmp_path, config))]

    # A missing optional engraver is a property of the host: MIDI, tabs,
    # MusicXML and GP5 were all produced, so exiting 1 would break every caller
    # that never asked for a PDF.
    assert main(argv) == 0
    err = capsys.readouterr().err
    assert "note(s) about the arrangement" in err
    assert "stage failure(s)" not in err
    assert (out_dir / "score.musicxml").exists()
    assert not (out_dir / "score.pdf").exists()


def test_convert_with_a_crashing_musescore_exits_1(monkeypatch, tmp_path, capsys):
    from lyre.errors import PdfUnavailable

    def crashed(xml_path, pdf_path):
        raise PdfUnavailable("mscore failed with exit code 1: boom")

    config = _pipeline_config()
    argv, out_dir = _cli_pipeline(monkeypatch, tmp_path, pdf=crashed, config=config)
    argv += ["--checkpoint", str(_checkpoint(tmp_path, config))]

    # MuseScore ran and did not produce the PDF the user asked for.
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "boom" in err
    assert "1 stage failure(s)" in err


def test_convert_with_a_checkpoint_and_a_working_engraver_exits_0(
    monkeypatch, tmp_path
):
    config = _pipeline_config()
    argv, out_dir = _cli_pipeline(monkeypatch, tmp_path, config=config)
    argv += ["--checkpoint", str(_checkpoint(tmp_path, config))]
    assert main(argv) == 0
    assert (out_dir / "score.pdf").exists()
