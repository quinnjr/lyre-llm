import os

import torch

from lyre.arranger import build_arrangement
from lyre.arranger.render import render_all_ascii, write_gp5, write_musicxml, write_pdf
from lyre.instruments import Instrument, program_for
from lyre.labeling.rules import label_tracks
from lyre.separator import Separator
from lyre.io.decode import load_audio
from lyre.tracking.hmm import frames_to_notes
from lyre.tracking.midi_io import merge_instruments
from lyre.transcriber.features import compute_features, frame_rate
from lyre.transcriber.model import MultiPitchNet, predict_track


def _notes_from_stem(model, waveform, sample_rate, feats, tracking, device):
    freq = frame_rate(sample_rate, feats["hop_ms"])
    frame_sec = 1.0 / freq
    mono = waveform.mean(dim=0)
    logmel = compute_features(
        mono,
        sample_rate=sample_rate,
        n_mels=feats["n_mels"],
        n_fft=feats["n_fft"],
        f_min=feats["f_min"],
        f_max=feats["f_max"],
        hop_ms=feats["hop_ms"],
    )
    pitch, onset = predict_track(
        model,
        logmel,
        window_frames=feats["window_frames"],
        overlap=tracking.pop("window_overlap", 0.0),
        device=device,
    )
    params = {k: tracking[k] for k in ("min_note_sec", "merge_gap_sec") if k in tracking}
    return frames_to_notes(pitch, onset, frame_sec=frame_sec, **params)


def _make_instrument(name, notes):
    inst = Instrument(name=name, notes=notes, source=name)
    inst.program = program_for(name)
    if name == "drums":
        inst.is_drum = True
    return inst


class Converter:
    def __init__(
        self,
        config,
        checkpoint=None,
        device="auto",
        use_llm=False,
        model_name="htdemucs_6s",
    ):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and device == "auto" else device
        self.use_llm = use_llm
        self.separator = Separator(model_name=model_name, device=self.device)
        self.model = MultiPitchNet(
            n_mels=config["features"]["n_mels"],
            n_notes=config["model"]["n_notes"],
            channels=config["model"]["channels"],
        )
        if checkpoint and os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state)
        self.model.eval()

    def _stem_wavs(self, waveform, sample_rate):
        stems = self.separator.separate(waveform, sample_rate)
        return {name: wav for name, wav in stems.items() if wav.abs().max() > 1e-6}

    def transcribe(self, waveform, sample_rate):
        feats = self.config["features"]
        tracking = dict(self.config["tracking"])
        instruments = []
        for name, wav in self._stem_wavs(waveform, sample_rate).items():
            notes = _notes_from_stem(self.model, wav, sample_rate, feats, dict(tracking), self.device)
            instruments.append(_make_instrument(name, notes))
        return instruments

    def convert(self, source_path, out_dir, name="score", no_separate=False):
        os.makedirs(out_dir, exist_ok=True)
        audio = self.config["audio"]
        waveform, sample_rate = load_audio(
            source_path, sample_rate=audio["sample_rate"], channels=audio["decode_channels"]
        )
        if no_separate:
            base = _notes_from_stem(
                self.model, waveform, sample_rate,
                self.config["features"], dict(self.config["tracking"]), self.device,
            )
            instruments = [_make_instrument("other", base)]
        else:
            instruments = self.transcribe(waveform, sample_rate)

        llm = None
        if self.use_llm:
            try:
                from lyre.labeling.llm import llm_from_env

                llm = llm_from_env()
            except Exception:
                llm = None
        instruments = label_tracks(instruments, llm=llm)

        arrange_cfg = self.config.get("arrange", {})
        tempo = arrange_cfg.get("tempo", 120.0)
        ts = list(arrange_cfg.get("time_signature", [4, 4]))
        arrangement = build_arrangement(
            instruments,
            tempo=tempo,
            time_signature=ts,
            guitar_tuning=arrange_cfg.get("guitar_tuning"),
            bass_tuning=arrange_cfg.get("bass_tuning"),
        )

        merge_instruments(instruments, tempo=tempo, time_signature=ts).write(
            os.path.join(out_dir, f"{name}.mid")
        )
        for inst in instruments:
            if inst.notes:
                merge_instruments([inst], tempo=tempo, time_signature=ts).write(
                    os.path.join(out_dir, f"{name}-{inst.name.replace(' ', '_')}.mid")
                )

        guitar_text, bass_text, drums_text = _split_ascii(render_all_ascii(arrangement))
        with open(os.path.join(out_dir, "guitar.tab"), "w") as fh:
            fh.write(guitar_text)
        with open(os.path.join(out_dir, "bass.tab"), "w") as fh:
            fh.write(bass_text)
        with open(os.path.join(out_dir, "drums.txt"), "w") as fh:
            fh.write(drums_text)

        write_musicxml(arrangement, os.path.join(out_dir, f"{name}.musicxml"))
        write_gp5(arrangement, os.path.join(out_dir, f"{name}.gp5"))
        try:
            write_pdf(os.path.join(out_dir, f"{name}.musicxml"), os.path.join(out_dir, f"{name}.pdf"))
        except Exception:
            pass

        return {
            "out_dir": out_dir,
            "instruments": [i.name for i in instruments],
            "files": sorted(os.listdir(out_dir)),
        }


def _split_ascii(text):
    sections = text.split("\n\n\n")
    guitar = "\n".join(part for part in sections if "Guitar" in part) if any("Guitar" in p for p in sections) else text
    bass = "\n".join(part for part in sections if "Bass" in part)
    drums = "\n".join(part for part in sections if "Drums" in part)
    return guitar.strip(), bass.strip(), drums.strip()