import torch

from lyre.transcriber.features import compute_features, frame_rate
from lyre.transcriber.model import MultiPitchNet


def test_frame_rate():
    assert frame_rate(44100, hop_ms=10) == 100.0
    assert frame_rate(44100, hop_ms=1) > 1000.0


def test_compute_features_shape():
    sr = 44100
    wav = torch.randn(sr)  # 1s
    mel = compute_features(wav, sample_rate=sr, n_mels=229, hop_ms=10)
    frames = mel.shape[0]
    assert mel.ndim == 2
    assert mel.shape[1] == 229
    assert frames == 100 + 1


def test_model_output_shapes():
    model = MultiPitchNet(n_mels=229, n_notes=128, channels=(32, 64, 96, 128))
    model.eval()
    x = torch.randn(2, 1, 128, 229)
    with torch.no_grad():
        pitch, onset = model(x)
    assert pitch.shape == (2, 128, 128)
    assert onset.shape == (2, 128, 1)


def test_model_param_count_small():
    model = MultiPitchNet(n_mels=229, n_notes=128, channels=(32, 64, 96, 128))
    n = sum(p.numel() for p in model.parameters())
    assert n < 30_000_000