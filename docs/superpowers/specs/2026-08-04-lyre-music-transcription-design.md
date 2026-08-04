# Lyre — Audio-to-MIDI & Guitar Tab Transcription Design

Date: 2026-08-04
Status: Approved (design review)

## Purpose

Lyre is a small-model pipeline that converts WAV/FLAC/MP3/MP4 audio into
multitrack MIDI, Guitar Pro tablature, and human-readable guitar/bass/drum
sheets — with the ability to distinguish every instrument (voice) on the
track.

## Architecture

```
 audio (wav/flac/mp3/mp4)
      │
   [decode]  ffmpeg → 44.1kHz PCM
      │
   [Stage 1: separation]  Demucs htdemucs_6s (FROZEN, no training)
      │  → 6 stems: bass, drums, other, vocals, guitar, piano
      │
   [Stage 2: multi-pitch CNN]  per-stem, ONE shared model (instrument-agnostic)
      │  log-mel spectrogram → pitch-activity (128 MIDI pitches/frame) + onset probs
      │
   [Stage 3: note tracking]  HMM/peak-picking, deterministic, no learning
      │  frames → note events (onset, pitch, velocity, offset)
      │
   [merge]  per-stem MIDI → multitrack MIDI (one track per stem)
      │
   [Stage 4: arrange-to-tab]  deterministic, no learning
      │  polyphony reduction → guitar/bass/drum parts → voicings → notation
      │
      ▼
   out: score.mid, per-stem .mid, guitar.tab, bass.tab, drums.txt,
        score.gp5, score.musicxml, score.pdf
```

### Stage roles

- **Stage 1 — Source separation (frozen).** `htdemucs_6s` is chosen specifically
  because it emits dedicated `guitar` and `piano` stems alongside bass, drums,
  vocals, and other — exactly the separation we need for tab quality. We only
  write a wrapper plus a separation eval; no training.
- **Stage 2 — Transcription model (the only trained component).** One shared
  residual CNN (~15–30M params) trained across all stems at once. Instrument
  distinction is handled by the stems; the model is instrument-agnostic. Input:
  log-mel spectrogram (128 frames × ~229 mels, 10ms hop). Output per frame:
  128-dim pitch-activity map + onset probability (+ optional velocity head).
  Later fine-tuned on guitar-heavy data so guitar transcription exceeds
  average-stem quality.
- **Stage 3 — Note tracking (deterministic).** HMM decoder (Basic Pitch-style)
  with a per-stem sustain prior, configurable and eval-gated. Converts frame
  predictions into MIDI note events.
- **Stage 4 — Arrange-to-tab (deterministic).** Reduces the multitrack MIDI to
  playable parts and renders them.
  - Guitar: group simultaneous notes → chords → voice into standard EADGBE
    within max fret reach (default ≤4-fret span, ≤24th fret), minimize
    voice-leading, keep lead melody on top when playable.
  - Bass: bass stem → 4-string EADG tab with position markers.
  - Drums: drum stem pitches mapped via GM kit → kick/snare/hat grid.
  - Rendering: ASCII `.tab`/`.txt`, Guitar Pro `.gp5` (via `PyGuitarPro`),
    MusicXML (universal interchange, tab staves), PDF (headless, via the
    MuseScore CLI).
    > **Amended 2026-08-04 (audit).** Originally "Verovio primary, MuseScore CLI
    > fallback". The Verovio Python toolkit renders SVG/MIDI/timemap and has no
    > PDF writer, so the primary path could never succeed. MuseScore is now the
    > only PDF renderer; PDF is unavailable when no MuseScore binary is on PATH,
    > and that is reported rather than swallowed.

## Training regimen (Stage 2)

### Data pipeline

- Input: 44.1kHz mono stem audio → log-mel spectrogram, 229 mels,
  128-frame (~1.28s) windows, 10ms frame hop.
- Targets derived from aligned MIDI per stem: frame-level pitch-activity
  (MIDI range ~24–95, covering bass low E up through guitar), onset frames,
  velocity.
- Datasets (rendered to stem audio + ground-truth MIDI):
  - **Slakh2100** — multi-instrument; bass, drums, guitar, piano, other. Primary.
  - **MAESTRO** — piano stems (high note-accuracy ground truth).
  - **GuitarSet** — guitar stem with aligned MIDI (guitar-specific data).
  - **Private corpus** — normalized into the same stem-audio + MIDI format.
- Data mix sampling weights: Slakh 40% / GuitarSet 25% / MAESTRO (piano) 25% /
  private 10% — the four `data.mix_weights` keys, named for the corpus rather
  than the instrument so they match the source tag prepare-data stamps.
  Guitar and piano are deliberately oversampled because tabs live or die there;
  bass and drums get their own dedicated share so no instrument is weak.

### Training

- Model: small residual CNN encoder + frame classifier head (~15–30M),
  shared/instrument-agnostic.
- Loss: BCE(pitch) + λ·BCE(onset); velocity L1 added once note tracking is clean.
- Optimizer AdamW, lr 1e-3, cosine decay, warmup, batch 32–64, mixed precision.
- Augmentation: pitch shift, time stretch, gain, SpecAugment, and re-mixing of
  stems into new mixes (also provides separator test material).
  > **Amended 2026-08-04 (audit).** Pitch shift, time stretch, gain and
  > SpecAugment are implemented behind the `augment:` config block. **Stem
  > re-mixing is NOT implemented** — it needs cross-track sampling that cannot
  > live in `Dataset.__getitem__`, and remains an open item.
- Phase 1: pretrain on all stems. Phase 2: fine-tune on guitar-heavy mix
  (GuitarSet + private guitar).
- Reproducibility: fixed splits, seeded runs, YAML config per run, metrics +
  checkpoints (best-F1 / latest / guitar-finetuned) logged per run.

### Eval harness

Runs on both raw stems and the full Demucs pipeline (to measure real-world
degradation):

- Frame-level multi-pitch F1 (with tolerance) and onset F1.
- Note-level onset/offset F1 via `mir_eval.transcription`.
- Per-instrument breakdown: bass, drums, guitar, piano, vocals, other — the
  "no weak voice" gate.
- Guitar tab quality proxy: note accuracy on GuitarSet test + fret-playability
  checks.

## Product skeleton

Python package `lyre/` with modules:

- `io/` — ffmpeg decode, format wrangling
- `separator/` — Demucs wrapper
- `transcriber/` — model, data pipeline, training
- `tracking/` — HMM note tracking
- `arranger/` — guitar/bass/drum arrangement + renderers (ASCII, GP5, MusicXML, PDF)
- `cli/` — command-line interface
- `configs/` — YAML run configs
- `scripts/` — data prep, eval, export

CLI: `lyre convert song.mp3 -o out/` → the full output bundle.

Model export: ONNX + optional int8 quantization of the transcription model;
optional FastAPI endpoint wrapping the same pipeline.

## Error handling

- Corrupt/undecodable audio → clean error, skip, non-zero exit code.
- Stem with nothing detected → empty-track warning, still merges.
- Guitar part unplayable (above fret 24) → octave shift, noted in output.
- Long files → chunked streaming to bound memory.
  > **Amended 2026-08-04 (audit).** Separation and transcription are chunked into
  > `inference.chunk_sec` blocks with `chunk_pad_sec` context, so peak stem memory
  > is bounded. **The initial decode is still full-file** — `load_audio` reads the
  > whole waveform into memory, and there is no streaming-decode hook of any kind:
  > adding one means writing a chunked decoder, not wiring up existing parameters.
- Stage failures isolated — a failed stage is reported clearly while remaining
  outputs are still produced.

## Testing

- Unit tests per stage: mel extraction, HMM note tracking, chord voicing, tab
  rendering, data transforms, config validation.
- Golden tests: synthetic audio → known MIDI → known tab byte-compare.
- Pipeline smoke: tiny checkpoint + real CLI end-to-end run in CI.
- Eval harness (above) doubles as the regression gate.
- pytest + CI running unit and smoke tests.

## Non-goals (first cut)

- Training a custom source separator (Demucs stays frozen).
- Event-token (MT3-style) transcription — documented as an upgrade path if
  frame-level metrics plateau.
- Effect decoding (bends, slides) beyond basic notation.
