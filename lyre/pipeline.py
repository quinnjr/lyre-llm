# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import os
import tempfile

import torch

from lyre.arranger import build_arrangement
from lyre.arranger.render import render_parts, write_gp5, write_musicxml, write_pdf
from lyre.errors import CheckpointNotFound, LyreError, PdfBackendMissing
from lyre.instruments import Instrument, program_for
from lyre.labeling.rules import label_tracks
from lyre.reporting import warn
from lyre.separator import Separator
from lyre.io.decode import load_audio
from lyre.tracking.hmm import Note, frames_to_notes, tracking_params
from lyre.tracking.midi_io import merge_instruments
from lyre.transcriber.features import compute_features, frame_rate
from lyre.transcriber.model import MultiPitchNet, predict_track

# Default block length for the chunked (bounded-memory) transcription path.
DEFAULT_CHUNK_SEC = 120.0
DEFAULT_CHUNK_PAD_SEC = 2.0

# Peak amplitude at or below which a stem is treated as containing nothing.
# Demucs returns near-zero rather than exactly-zero for an absent instrument.
SILENCE_FLOOR = 1e-6

# The ASCII bundle is one stage writing three files. Both the writer and the
# stage's declared output list read this: a hand-maintained second copy drifts,
# and the drift is invisible until a failure cleanup misses a file.
ASCII_BUNDLE = ("guitar.tab", "bass.tab", "drums.txt")


def _is_silent(wav):
    return wav.numel() == 0 or float(wav.abs().max()) <= SILENCE_FLOOR


def _fingerprint(path):
    """Identity of a file for "did this run touch it?", or None if absent."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _remove_touched(paths, before):
    """Delete only the paths this run created or replaced."""
    for path in paths:
        now = _fingerprint(path)
        if now is None or now == before.get(path):
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def _config_float(name, value):
    """Coerce a config number, reporting a bad one the way every other
    configuration mistake is reported: one line, not a ``float()`` traceback."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        raise LyreError(f"{name} must be a number, got {value!r}") from None


def _silent_everywhere(blocks):
    """Stem names that were silent in every separated block, sorted."""
    if not blocks:
        return []
    common = set(blocks[0])
    for block in blocks[1:]:
        common &= block
    return sorted(common)


def _atomic_write(path, write):
    """Write ``path`` through a temporary sibling, renamed into place at the end.

    A writer that dies part-way through would otherwise leave a truncated file
    that every later existence check — including the PDF stage's — treats as a
    real output.

    The scratch file is created by ``mkstemp`` rather than named ``path +
    ".tmp"``: the predictable name is guessable by another user in a shared
    output directory, and two concurrent runs writing the same score would
    otherwise share one scratch file and interleave their bytes.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".lyre-", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        # A failure to clean up scratch must never replace the real exception
        # with a confusing one about a temp file nobody asked about.
        if os.path.lexists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _notes_from_stem(model, waveform, sample_rate, feats, tracking, device, overlap=0.5):
    freq = frame_rate(sample_rate, feats["hop_ms"])
    frame_sec = 1.0 / freq
    mono = waveform.mean(dim=0) if waveform.ndim > 1 else waveform
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
        overlap=overlap,
        device=device,
    )
    return frames_to_notes(pitch, onset, frame_sec=frame_sec, **tracking_params(tracking))


def _make_instrument(name, notes):
    inst = Instrument(name=name, notes=notes, source=name)
    inst.program = program_for(name)
    if name == "drums":
        inst.is_drum = True
    return inst


def _shift_notes(notes, offset, keep_lo, keep_hi, clip_end=None):
    """Offset a block's notes to absolute time and keep only its core window."""
    out = []
    for n in notes:
        start = n.start + offset
        if start < keep_lo - 1e-9 or start >= keep_hi - 1e-9:
            continue
        end = n.end + offset
        if clip_end is not None:
            end = min(end, clip_end)
        out.append(Note(pitch=n.pitch, start=start, end=max(end, start), velocity=n.velocity))
    return out


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
        # Problems found while building the converter, replayed onto `convert`'s
        # failure channel: __init__ has no channel of its own to report into.
        self._startup_failures = []
        self.device = "cuda" if torch.cuda.is_available() and device == "auto" else device
        # Either the --llm flag or the ``labeling.use_llm`` config key turns it
        # on; the flag can only add, never override a config that enabled it.
        self.use_llm = bool(use_llm) or bool(config.get("labeling", {}).get("use_llm", False))
        # Every knob is read and bounded here, before anything expensive runs.
        # Validating them lazily meant a bad value surfaced after separation and
        # transcription had already burned minutes of GPU time — or, for a
        # negative chunk length, never surfaced at all because the block loop
        # stopped making progress.
        self._overlap()
        self._chunk_sec()
        self._chunk_pad_sec()
        self.llm = None
        if self.use_llm:
            from lyre.labeling.llm import llm_from_env

            # A user who explicitly asked for the LLM gets a hard error, not a
            # silent downgrade to rule-based labels.
            self.llm = llm_from_env(required=True)
        self.separator = Separator(model_name=model_name, device=self.device)
        self.model = MultiPitchNet(
            n_mels=config["features"]["n_mels"],
            n_notes=config["model"]["n_notes"],
            channels=config["model"]["channels"],
        )
        if checkpoint:
            if not os.path.exists(checkpoint):
                raise CheckpointNotFound(
                    f"checkpoint not found: {checkpoint} — "
                    "refusing to transcribe with untrained weights"
                )
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state)
        else:
            # This is a failure of the run, not an advisory about the music:
            # untrained weights produce tabs that look plausible and mean
            # nothing, and `lyre convert && publish` must not treat them as a
            # result. It is stashed because neither channel exists yet.
            warn(
                "no checkpoint given — using untrained (randomly initialised) "
                "weights; the transcription will be meaningless",
                self._startup_failures,
            )
        self.model.eval()

    # ---- config helpers -------------------------------------------------

    def _overlap(self):
        """Fraction of each inference window shared with the next one.

        Read from ``inference.window_overlap``, not from ``tracking``: it is a
        property of how the model is run over the mel, not of note tracking.

        The value is bounded here because it feeds
        ``step = max(1, int(window_frames * (1 - overlap)))``: at 1.0 the step
        collapses to a single frame, which is ~128x the work and looks exactly
        like a hang rather than like a bad config value.
        """
        value = _config_float(
            "inference.window_overlap", self.config.get("inference", {}).get("window_overlap", 0.5)
        )
        if not 0.0 <= value < 1.0:
            raise LyreError(
                f"inference.window_overlap must be >= 0.0 and < 1.0, got {value}; "
                "1.0 would advance the analysis window one frame at a time and "
                "never finish"
            )
        return value

    def _chunk_sec(self):
        """Block length for the chunked path; 0 disables blocking.

        Negative is rejected rather than clamped: it makes each block's end fall
        before its start, so the loop's cursor moves backwards and the run never
        terminates.
        """
        value = _config_float(
            "inference.chunk_sec", self.config.get("inference", {}).get("chunk_sec", DEFAULT_CHUNK_SEC)
        )
        if value < 0.0:
            raise LyreError(f"inference.chunk_sec must be >= 0.0, got {value}")
        return value

    def _chunk_pad_sec(self):
        """Context decoded either side of a block and then discarded.

        Negative is rejected because it moves the block boundaries *inward*,
        leaving a band of audio around every seam that no block ever sees — a
        silent hole in the transcription rather than a visible error.
        """
        value = _config_float(
            "inference.chunk_pad_sec",
            self.config.get("inference", {}).get("chunk_pad_sec", DEFAULT_CHUNK_PAD_SEC),
        )
        if value < 0.0:
            raise LyreError(f"inference.chunk_pad_sec must be >= 0.0, got {value}")
        return value

    # ---- separation / transcription --------------------------------------

    def _stem_wavs(self, waveform, sample_rate, silent=None):
        """Separate into stems.

        A silent stem is kept rather than dropped so the instrument still shows
        up (empty) in the outputs. Which stems were silent is *recorded* here
        and reported by the caller: this runs once per block, and a stem that is
        silent in one block of a long track is not an empty instrument — saying
        so per block is both repetitive and untrue.
        """
        stems, stem_rate = self.separator.separate(waveform, sample_rate)
        if silent is not None:
            silent.append({name for name, wav in stems.items() if _is_silent(wav)})
        return stems, stem_rate

    def _block_notes(self, waveform, sample_rate, no_separate=False, silent=None):
        """Transcribe one waveform block -> ``{stem_name: [Note, ...]}``."""
        feats = self.config["features"]
        tracking = dict(self.config["tracking"])
        overlap = self._overlap()
        if no_separate:
            whole = _notes_from_stem(
                self.model, waveform, sample_rate, feats, tracking, self.device, overlap=overlap
            )
            return {"other": whole}
        stems, stem_rate = self._stem_wavs(waveform, sample_rate, silent=silent)
        out = {}
        for name in sorted(stems):
            wav = stems[name]
            if _is_silent(wav):
                out[name] = []
                continue
            out[name] = _notes_from_stem(
                self.model, wav, stem_rate, feats, tracking, self.device, overlap=overlap
            )
        return out

    def transcribe(self, waveform, sample_rate, no_separate=False, notes=None):
        """Transcribe a full waveform into labelled-by-stem instruments.

        Long inputs are processed in overlapping blocks so peak memory is
        bounded by the block length rather than by the length of the track.
        """
        total = float(waveform.shape[-1]) / float(sample_rate)
        chunk_sec = self._chunk_sec()
        silent = []
        if not chunk_sec or total <= chunk_sec:
            per_stem = self._block_notes(
                waveform, sample_rate, no_separate=no_separate, silent=silent
            )
        else:
            per_stem = self._chunked_notes(
                waveform, sample_rate, total, chunk_sec,
                no_separate=no_separate, silent=silent,
            )
        # One advisory per track, and only for a stem the separator found
        # nothing in anywhere: with htdemucs_6s most real inputs have at least
        # one, so this is information about the recording, not a failed run.
        for name in _silent_everywhere(silent):
            warn(
                f"stem '{name}': nothing detected (silent); merging as an empty track",
                notes,
            )
        return [_make_instrument(name, per_stem[name]) for name in sorted(per_stem)]

    def _chunked_notes(self, waveform, sample_rate, total, chunk_sec, no_separate=False, silent=None):
        pad = min(self._chunk_pad_sec(), chunk_sec / 4.0)
        per_stem = {}
        start = 0.0
        while start < total - 1e-6:
            end = min(start + chunk_sec, total)
            lo = max(0.0, start - pad)
            hi = min(total, end + pad)
            block = waveform[..., int(round(lo * sample_rate)) : int(round(hi * sample_rate))]
            if block.shape[-1] == 0:
                break
            block_notes = self._block_notes(
                block, sample_rate, no_separate=no_separate, silent=silent
            )
            keep_hi = end
            for name, notes in block_notes.items():
                per_stem.setdefault(name, []).extend(
                    _shift_notes(notes, lo, start, keep_hi, clip_end=total)
                )
            start = end
        for name in per_stem:
            per_stem[name].sort(key=lambda n: (n.start, n.pitch))
        return per_stem

    # ---- stage isolation --------------------------------------------------

    def _stage(self, name, fn, failures, paths=(), notes=None, advisory=()):
        """Run one output stage in isolation and report what it actually wrote.

        One renderer failing should not cost the user the other five outputs, so
        the failure is recorded and the pipeline continues.

        A failed stage must not leave a half-written file behind, but it must
        also not destroy an *earlier* run's good output: most stages write
        through ``_atomic_write``, so on failure the final path still holds
        whatever was there before this run started. Only paths whose identity
        changed during the stage are removed — which is exactly the PDF stage
        (MuseScore writes in place) and the ASCII bundle, where files 1..k-1
        were already replaced when file k failed.

        Exception types listed in ``advisory`` are recorded on ``notes``
        instead: a missing external engraver is a property of the host, not a
        run that failed to do what it was asked.

        Returns the list of paths this run wrote — never a directory listing,
        which cannot tell this run's output from last run's.
        """
        before = {path: _fingerprint(path) for path in paths}
        try:
            fn()
        except advisory as exc:
            warn(f"{name}: {exc}", failures if notes is None else notes)
            _remove_touched(paths, before)
            return []
        except Exception as exc:
            warn(f"{name}: {exc}", failures)
            _remove_touched(paths, before)
            return []
        made = [p for p in paths if os.path.exists(p)]
        missing = [p for p in paths if p not in made]
        if missing:
            # An external tool can exit 0 and write nothing. Reporting that as
            # success gives the user no error and no file.
            warn(
                "%s: reported success but did not write %s"
                % (name, ", ".join(os.path.basename(p) for p in missing)),
                failures,
            )
        return made

    # ---- main entry point --------------------------------------------------

    def convert(self, source_path, out_dir, name="score", no_separate=False):
        # Two separate channels, because they mean different things to a caller.
        # ``failures`` is "an output you asked for is missing" and is what the
        # CLI turns into a non-zero exit. ``notes`` is "here is something about
        # the music you should know" — a silent stem, a phrase the arranger had
        # to move an octave. Notes are routine on real input; treating them as
        # failures made every successful conversion exit 1.
        failures = list(self._startup_failures)
        notes = []
        os.makedirs(out_dir, exist_ok=True)
        audio = self.config["audio"]

        # Decode / separation / transcription are fatal: nothing downstream is
        # meaningful without notes, so these deliberately propagate.
        waveform, sample_rate = load_audio(
            source_path, sample_rate=audio["sample_rate"], channels=audio["decode_channels"]
        )
        instruments = self.transcribe(
            waveform, sample_rate, no_separate=no_separate, notes=notes
        )
        del waveform

        instruments = label_tracks(
            instruments, llm=self.llm, sink=notes, strict=self.use_llm
        )

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
        for message in _performance_notes(arrangement):
            warn(message, notes)

        written = []
        midi_path = os.path.join(out_dir, f"{name}.mid")
        written += self._stage(
            "midi merge",
            lambda: _atomic_write(
                midi_path,
                lambda tmp: merge_instruments(
                    instruments, tempo=tempo, time_signature=ts
                ).write(tmp),
            ),
            failures,
            paths=(midi_path,),
        )
        for inst in instruments:
            if not inst.notes:
                continue
            part_path = os.path.join(
                out_dir, f"{name}-{inst.name.replace(' ', '_')}.mid"
            )
            written += self._stage(
                f"midi merge ({inst.name})",
                lambda inst=inst, part_path=part_path: _atomic_write(
                    part_path,
                    lambda tmp: merge_instruments(
                        [inst], tempo=tempo, time_signature=ts
                    ).write(tmp),
                ),
                failures,
                paths=(part_path,),
            )

        tab_paths = tuple(os.path.join(out_dir, f) for f in ASCII_BUNDLE)
        written += self._stage(
            "ascii tabs",
            lambda: _write_ascii_bundle(arrangement, out_dir),
            failures,
            paths=tab_paths,
        )
        musicxml_path = os.path.join(out_dir, f"{name}.musicxml")
        written += self._stage(
            "musicxml",
            lambda: _atomic_write(
                musicxml_path, lambda tmp: write_musicxml(arrangement, tmp)
            ),
            failures,
            paths=(musicxml_path,),
        )
        gp5_path = os.path.join(out_dir, f"{name}.gp5")
        written += self._stage(
            "gp5",
            lambda: _atomic_write(gp5_path, lambda tmp: write_gp5(arrangement, tmp)),
            failures,
            paths=(gp5_path,),
        )
        # Engrave from the MusicXML *this* run produced. Testing for the file on
        # disk instead would happily engrave last run's score after the MusicXML
        # stage failed, and report both "musicxml failed" and "score.pdf" — the
        # PDF being of a different arrangement entirely.
        if musicxml_path in written:
            pdf_path = os.path.join(out_dir, f"{name}.pdf")
            # No MuseScore on this host is an environment property, not a run
            # that failed: MIDI, tabs, MusicXML and GP5 all succeeded, and
            # exiting 1 for a missing optional binary breaks every caller that
            # never asked for a PDF.
            written += self._stage(
                "pdf",
                lambda: write_pdf(musicxml_path, pdf_path),
                failures,
                paths=(pdf_path,),
                notes=notes,
                advisory=(PdfBackendMissing,),
            )
        else:
            warn("pdf: skipped because MusicXML was not written", failures)

        return {
            "out_dir": out_dir,
            "instruments": [i.name for i in instruments],
            "files": sorted(os.path.basename(p) for p in written),
            "failures": failures,
            "notes": notes,
        }


def _performance_notes(arrangement):
    """Surface per-track performance notes (octave shifts, dropped notes)."""
    messages = []
    for group in ("guitar", "bass"):
        for track in getattr(arrangement, group):
            for line in track.performance_notes:
                messages.append(f"{track.name}: {line}")
    return messages


def _write_ascii_bundle(arrangement, out_dir):
    for filename, text in zip(ASCII_BUNDLE, render_parts(arrangement)):
        _atomic_write(
            os.path.join(out_dir, filename),
            lambda tmp, text=text: _write_text(tmp, text),
        )


def _write_text(path, text):
    with open(path, "w") as fh:
        fh.write(text + "\n" if text else "")
