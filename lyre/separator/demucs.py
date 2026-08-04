import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


STEMS = ("bass", "drums", "guitar", "other", "piano", "vocals")


def _pick_device(device):
    if device in ("auto", None):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Separator:
    def __init__(self, model_name="htdemucs_6s", device="auto", shifts=0):
        self.model_name = model_name
        self.device = _pick_device(device)
        self.shifts = shifts
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    @property
    def samplerate(self):
        self._ensure_model()
        return self.model.samplerate

    @property
    def sources(self):
        self._ensure_model()
        return list(self.model.sources)

    def separate(self, waveform, sample_rate=44100):
        self._ensure_model()
        wav = waveform.to(self.device)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if sample_rate != self.model.samplerate:
            wav = torchaudio_resample(wav, sample_rate, self.model.samplerate)
        mix = wav.unsqueeze(0)
        with torch.no_grad():
            sources = apply_model(
                self.model,
                mix,
                device=self.device,
                shifts=self.shifts,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
        stems = {name: sources[i].cpu() for i, name in enumerate(self.model.sources)}
        for name in STEMS:
            if name not in stems:
                stems[name] = torch.zeros_like(wav.cpu())
        return stems


def torchaudio_resample(waveform, sr_in, sr_out):
    import torchaudio.functional as F

    return F.resample(waveform, sr_in, sr_out)
