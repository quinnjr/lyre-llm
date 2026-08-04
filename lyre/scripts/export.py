import torch
import yaml


def main(checkpoint, out, config_path):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    feats = config["features"]
    from lyre.transcriber.model import MultiPitchNet

    model = MultiPitchNet(
        n_mels=feats["n_mels"], n_notes=config["model"]["n_notes"], channels=config["model"]["channels"]
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    example = torch.randn(1, 1, feats["window_frames"], feats["n_mels"])
    torch.onnx.export(
        model,
        example,
        out,
        input_names=["logmel"],
        output_names=["pitch_logits", "onset_logits"],
        opset_version=17,
        dynamic_axes={"logmel": {2: "time"}},
    )
    print(f"exported -> {out}")