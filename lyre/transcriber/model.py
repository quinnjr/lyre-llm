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


class MultiPitchNet(nn.Module):
    def __init__(
        self,
        n_mels=229,
        n_notes=128,
        channels=(32, 64, 96, 128),
        n_onset_classes=1,
    ):
        super().__init__()
        self.n_notes = n_notes
        layers = [nn.Conv2d(1, channels[0], 3, padding=(1, 1)), nn.BatchNorm2d(channels[0]), nn.ReLU()]
        for i in range(1, len(channels)):
            layers.append(_Block(channels[i - 1], channels[i], stride=(1, 2)))
        self.encoder = nn.Sequential(*layers)
        self.head_pitch = nn.Conv2d(channels[-1], n_notes, 1)
        self.head_onset = nn.Conv2d(channels[-1], n_onset_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=3, keepdim=True)
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
    step = max(1, int(window_frames * (1 - overlap)))
    pitch_acc = torch.zeros(total, model.n_notes, device=device)
    onset_acc = torch.zeros(total, 1, device=device)
    count = torch.zeros(total, 1, device=device)
    starts = list(range(0, max(1, total - window_frames + 1), step))
    if not starts:
        starts = [0]
    for start in starts:
        end = min(start + window_frames, total)
        start = end - window_frames
        seg = logmel[start:end].unsqueeze(0).unsqueeze(0).to(device)
        pitch, onset = model(seg)
        pitch = torch.sigmoid(pitch)
        onset = torch.sigmoid(onset)
        pitch_acc[start:end] += pitch[0]
        onset_acc[start:end] += onset[0]
        count[start:end] += 1
    pitch = (pitch_acc / count.clamp_min(1)).cpu()
    onset = (onset_acc / count.clamp_min(1)).cpu()
    return pitch, onset
