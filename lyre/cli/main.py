import argparse
import sys

import yaml

from lyre.pipeline import Converter


def _load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def cmd_convert(args):
    config = _load_config(args.config)
    conv = Converter(
        config,
        checkpoint=args.checkpoint,
        device=args.device,
        use_llm=args.llm,
        model_name=args.model,
    )
    result = conv.convert(args.audio, args.out, name=args.name, no_separate=args.no_separate)
    sys.stdout.write("instruments: %s\n" % ", ".join(result["instruments"]))
    for f in result["files"]:
        sys.stdout.write("  %s\n" % f)


def cmd_train(args):
    from lyre.scripts.train import main as train_main

    train_main(args.config, args.checkpoint, args.devices, args.epochs, args.resume)


def cmd_eval(args):
    from lyre.scripts.evaluate import main as eval_main

    eval_main(args.config, args.checkpoint, args.index, args.device)


def cmd_prepare(args):
    from lyre.scripts.prepare_data import main as prepare_main

    prepare_main(args.config)


def cmd_export(args):
    from lyre.scripts.export import main as export_main

    export_main(args.checkpoint, args.out, args.config)


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
    c.set_defaults(func=cmd_eval)

    c = sub.add_parser("prepare-data", help="scan datasets and build training indexes")
    c.add_argument("--config", default="configs/default.yaml")
    c.set_defaults(func=cmd_prepare)

    c = sub.add_parser("export", help="export transcription model to ONNX")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("-o", "--out", default="model.onnx")
    c.add_argument("--config", default="configs/default.yaml")
    c.set_defaults(func=cmd_export)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)