import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        shortcut = []
        if stride != (1, 1) or in_channels != out_channels:
            shortcut.append(nn.Conv2d(in_channels, out_channels, 1, stride=stride))
        self.shortcut = nn.Sequential(*shortcut) if shortcut else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def _freq_out(n_mels, n_strided):
    """Frequency width after ``n_strided`` stride-2 3x3 convs (padding 1)."""
    size = int(n_mels)
    for _ in range(n_strided):
        size = (size - 1) // 2 + 1
    return size


class MultiPitchNet(nn.Module):
    def __init__(
        self,
        n_mels=229,
        n_notes=128,
        channels=(128, 256, 512, 896),
        n_onset_classes=1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_notes = n_notes
        layers = [nn.Conv2d(1, channels[0], 3, padding=(1, 1)), nn.BatchNorm2d(channels[0]), nn.ReLU()]
        for i in range(1, len(channels)):
            layers.append(_Block(channels[i - 1], channels[i], stride=(1, 2)))
        self.encoder = nn.Sequential(*layers)
        self.freq_out = _freq_out(n_mels, len(channels) - 1)
        # Learned collapse of the (reduced) frequency axis; replaces a plain mean
        # and is what makes n_mels load-bearing.
        self.freq_pool = nn.Linear(self.freq_out, 1)
        self.head_pitch = nn.Conv2d(channels[-1], n_notes, 1)
        self.head_onset = nn.Conv2d(channels[-1], n_onset_classes, 1)

    def forward(self, x):
        if x.shape[-1] != self.n_mels:
            raise ValueError(
                f"expected {self.n_mels} mel bins on the last axis, got {x.shape[-1]}"
            )
        x = self.encoder(x)
        x = self.freq_pool(x)
        pitch = self.head_pitch(x).squeeze(3).transpose(1, 2)
        onset = self.head_onset(x).squeeze(3).transpose(1, 2)
        return pitch, onset


@torch.inference_mode()
def predict_track(
    model,
    logmel,
    window_frames=128,
    overlap=0.5,
    device="cpu",
):
    model.eval()
    model.to(device)
    total = logmel.shape[0]
    if total == 0:
        raise ValueError("empty log-mel input")
    step = max(1, int(window_frames * (1 - overlap)))
    pitch_acc = torch.zeros(total, model.n_notes, device=device)
    onset_acc = torch.zeros(total, 1, device=device)
    count = torch.zeros(total, 1, device=device)
    if total <= window_frames:
        starts = [0]
    else:
        starts = list(range(0, total - window_frames + 1, step))
        # Cover the tail: without this the last (total - window_frames) % step
        # frames are never fed to the model and silently emit probability 0.
        if (total - window_frames) % step != 0:
            starts.append(total - window_frames)
    for start in starts:
        end = min(start + window_frames, total)
        seg = logmel[start:end]
        n = seg.shape[0]
        if n < window_frames:
            seg = torch.nn.functional.pad(seg, (0, 0, 0, window_frames - n), value=float(seg.min()))
        seg = seg.unsqueeze(0).unsqueeze(0).to(device)
        pitch, onset = model(seg)
        pitch = torch.sigmoid(pitch)
        onset = torch.sigmoid(onset)
        pitch_acc[start:end] += pitch[0, :n]
        onset_acc[start:end] += onset[0, :n]
        count[start:end] += 1
    if float(count.min()) <= 0:
        raise RuntimeError("predict_track left frames uncovered")
    pitch = (pitch_acc / count).cpu()
    onset = (onset_acc / count).cpu()
    return pitch, onset
