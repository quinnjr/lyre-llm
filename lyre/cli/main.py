import argparse
import os
import sys

from lyre.config import load_config
from lyre.errors import LyreError
from lyre.pipeline import Converter


def _exit_code(result):
    """Map a script's result dict to a process exit code.

    Only ``failures`` — "something you asked for did not happen" — is non-zero.
    Advisory output lives under ``notes`` and never affects the exit code, so
    this deliberately looks at one key and no other: honouring a second,
    loosely-named list is how "any warning means exit 1" would come back.

    A script that returns nothing at all is treated as success, but it must
    never be *assumed* to have succeeded: this is why every ``cmd_*`` returns an
    int rather than falling off the end.
    """
    if isinstance(result, dict) and result.get("failures"):
        return 1
    return 0


def _checked_name(name):
    """Reject an output basename that could escape the output directory.

    ``name`` is interpolated into five output paths and those paths are deleted
    when a stage fails, so a separator or ``..`` in it would let
    ``--name ../../elsewhere/thing`` write and unlink outside ``--out``.
    """
    bad = [os.sep, "..", "\0"]
    if os.altsep:
        bad.append(os.altsep)
    for token in bad:
        if token in name:
            raise LyreError(
                f"--name must be a plain filename without {token!r}, got {name!r}"
            )
    if not name.strip():
        raise LyreError("--name must not be empty")
    return name


def cmd_convert(args):
    name = _checked_name(args.name)
    config = load_config(args.config)
    conv = Converter(
        config,
        checkpoint=args.checkpoint,
        device=args.device,
        use_llm=args.llm,
        model_name=args.model,
    )
    result = conv.convert(args.audio, args.out, name=name, no_separate=args.no_separate)
    sys.stdout.write("instruments: %s\n" % ", ".join(result["instruments"]))
    for f in result["files"]:
        sys.stdout.write("  %s\n" % f)
    # Failures and notes were already streamed to stderr as they happened, so
    # they are summarised here rather than printed a second time. Only stage
    # failures affect the exit code: an advisory note (a silent stem, an octave
    # shift the arranger had to make) accompanies a perfectly good conversion,
    # and exiting non-zero for one breaks every `set -e` caller.
    failures = result.get("failures") or []
    notes = result.get("notes") or []
    if notes:
        sys.stderr.write("%d note(s) about the arrangement (above)\n" % len(notes))
    if failures:
        sys.stderr.write(
            "%d stage failure(s); those outputs were not written\n" % len(failures)
        )
        return 1
    return 0


def cmd_train(args):
    from lyre.scripts.train import main as train_main

    result = train_main(args.config, args.checkpoint, args.devices, args.epochs, args.resume)
    return _exit_code(result)


def cmd_eval(args):
    from lyre.scripts.evaluate import main as eval_main

    result = eval_main(
        args.config,
        args.checkpoint,
        args.index,
        args.device,
        args.through_separator,
        args.separator_model,
    )
    return _exit_code(result)


def cmd_prepare(args):
    from lyre.scripts.prepare_data import main as prepare_main

    return _exit_code(prepare_main(args.config))


def cmd_export(args):
    from lyre.scripts.export import main as export_main

    return _exit_code(export_main(args.checkpoint, args.out, args.config, args.quantize))


def build_parser():
    p = argparse.ArgumentParser(prog="lyre")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="audio -> MIDI + tabs")
    c.add_argument("audio")
    c.add_argument("-o", "--out", default="out")
    c.add_argument("--config", default="configs/default.yaml")
    c.add_argument("--checkpoint", default=None)
    c.add_argument("--model", default="htdemucs_6s")
    c.add_argument("--device", default="auto")
    c.add_argument("--name", default="score")
    c.add_argument("--llm", action="store_true", help="use LLM registry labeling (LYRE_LLM_*)")
    c.add_argument("--no-separate", action="store_true", help="skip Demucs; treat whole file as one stem")
    c.set_defaults(func=cmd_convert)

    c = sub.add_parser("train", help="train / fine-tune the transcription model")
    c.add_argument("--config", default="configs/default.yaml")
    c.add_argument("--checkpoint", default=None)
    c.add_argument("--devices", default="auto")
    c.add_argument("--epochs", type=int, default=None)
    c.add_argument("--resume", default=None)
    c.set_defaults(func=cmd_train)

    c = sub.add_parser("eval", help="run transcription eval harness")
    c.add_argument("--config", default="configs/default.yaml")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--index", default=None)
    c.add_argument("--device", default="auto")
    c.add_argument(
        "--through-separator",
        action="store_true",
        help="route eval audio through Demucs first (real-world degradation)",
    )
    c.add_argument("--separator-model", default="htdemucs_6s")
    c.set_defaults(func=cmd_eval)

    c = sub.add_parser("prepare-data", help="scan datasets and build training indexes")
    c.add_argument("--config", default="configs/default.yaml")
    c.set_defaults(func=cmd_prepare)

    c = sub.add_parser("export", help="export transcription model to ONNX")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("-o", "--out", default="model.onnx")
    c.add_argument("--config", default="configs/default.yaml")
    c.add_argument(
        "--int8",
        dest="quantize",
        action="store_true",
        help="also write <out>.int8.onnx via dynamic quantization",
    )
    c.set_defaults(func=cmd_export)

    return p


def main(argv=None):
    """Run the CLI. Returns a process exit code (0 ok, 1 degraded, 2 error)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except LyreError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 2
    except FileNotFoundError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("error: interrupted\n")
        return 130
    return 0 if code is None else int(code)


if __name__ == "__main__":
    sys.exit(main())