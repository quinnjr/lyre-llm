# lyre

Audio → multitrack MIDI, Guitar Pro tablature, and readable guitar/bass/drum
sheets, with each instrument on its own track.

> **Status: the model has never been trained.**
>
> Every deterministic stage — separation wiring, note tracking, arrangement,
> voicing, and all five output formats — is implemented and tested. The neural
> stage is not: there is no checkpoint, no trained weights, and no measured
> accuracy. `lyre convert` without `--checkpoint` will run end to end and
> produce a complete output bundle from **randomly initialised weights**, which
> is musically meaningless. It says so on stderr and exits non-zero.
>
> Treat this as a working pipeline waiting for a model, not a working
> transcriber.

## How it works

```
audio (wav/flac/mp3/mp4)
  │
  ├─ decode ──────────── 44.1 kHz PCM
  ├─ separate ────────── Demucs htdemucs_6s, frozen: bass drums other vocals guitar piano
  ├─ transcribe ──────── one shared instrument-agnostic CNN (~16.5M params)
  │                      log-mel → 128 pitch activations + onset, per frame
  ├─ track ───────────── HMM decode, deterministic → note events
  ├─ label ───────────── rules, or an LLM if you point one at it
  └─ arrange ─────────── polyphony reduction → voicings → notation
         ↓
   score.mid · per-instrument .mid · guitar.tab · bass.tab · drums.txt
   score.gp5 · score.musicxml · score.pdf
```

Instrument identity comes from the stems, not from the model — so the same
network transcribes every instrument, and multi-guitar arrangements work by
splitting a guitar stem into rhythm and lead parts.

## Install

Python 3.10+. `torch` and `torchaudio` are heavy; install them however suits
your platform first.

```bash
pip install -e .
pip install -e '.[test]'      # pytest
pip install -e '.[export]'    # onnx, onnxscript, onnxruntime — only for `lyre export`
```

PDF export shells out to the **MuseScore CLI** (`mscore`/`musescore`), an
external binary rather than a Python package. Without it every other output is
still produced and the missing PDF is reported as a note, not a failure.

## Use

```bash
lyre convert song.mp3 -o out/ --checkpoint runs/default/best.pt
```

| Command | What it does |
|---|---|
| `convert` | audio → the full output bundle |
| `prepare-data` | scan corpora, build train/val/test indexes |
| `train` | train, or fine-tune from a checkpoint |
| `eval` | frame/onset/note F1, per instrument, plus a fret-playability proxy |
| `export` | ONNX, optionally int8 (`--int8`) |

Useful `convert` flags: `--no-separate` skips Demucs and treats the file as one
stem; `--llm` labels tracks with an LLM instead of the built-in rules (needs
`LYRE_LLM_ENDPOINT`, and refuses a plaintext endpoint); `--name` sets the output
basename.

**Exit codes.** `0` success, `1` a stage failed and its output is missing,
`2` bad input or config, `130` interrupted. A conversion that merely produced
advisory notes — a silent stem, an octave shift the arranger had to make, no
MuseScore installed — exits `0`.

## Configuration

`configs/default.yaml` is the full surface: run, audio, features, model,
augment, data, train, tracking, arrange, labeling, inference, eval.
`configs/guitar_finetune.yaml` shows the intended layering — `extends:
default.yaml` plus only what genuinely differs.

A test asserts every key in the config is read by real code and that every key
the code reads is declared. That gate exists because six advertised options
once did nothing.

## Training

The design targets Slakh2100 / MAESTRO / GuitarSet plus a private corpus, mixed
40/25/25/10 with guitar and piano oversampled. `data.sources` in the shipped
config holds **placeholder paths** — point them at real corpora before running
`prepare-data`.

`train` requires a GPU. `--checkpoint` loads weights and fine-tunes from epoch
0; `--resume` continues a run, restoring optimizer state, LR step and best
score.

## Development

```bash
pytest tests -q          # 630 tests, ~4s, no network
```

The suite is value-based on purpose. An earlier version asserted shapes and
non-emptiness, and passed while the model was 60× too small and every GP5
exported at 4× speed. Tests here assert reconstructed pitches, that measures
sum to their time signature, and that reverting a fix makes them fail.

Not exercised anywhere: real Demucs separation (needs a weights download), ONNX
export, and MuseScore rendering. Their contracts are tested against fakes; the
integrations themselves are unverified.

Design rationale, including three amendments where behaviour was reduced rather
than fixed, is in
[`docs/superpowers/specs/`](docs/superpowers/specs/2026-08-04-lyre-music-transcription-design.md).

## License

[Elastic License 2.0](LICENSE.txt) — source-available, not open source. Use,
modify and redistribute freely, including commercially and internally. You may
not offer lyre to third parties as a hosted or managed service.
