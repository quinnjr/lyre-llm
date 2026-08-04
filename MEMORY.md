# lyre — handoff memory

Audio → multitrack MIDI + guitar/bass/drums tab transcriber. Small-model pipeline,
per architecture spec in `docs/superpowers/specs/2026-08-04-lyre-music-transcription-design.md`.

## Objective & architecture (approved)
Frozen `Demucs htdemucs_6s` stem separation → **one shared instrument-agnostic residual CNN**
multi-pitch/onset frame detector (~15–30M params) → deterministic HMM note tracking →
deterministic arrange-to-tab.

- **Training mix (sampling weights)**: Slakh2100 40% / guitar (GuitarSet) 25% / piano (MAESTRO) 25% / private 10%.
  Loss = `BCE(pitch) + λ·BCE(onset)`; AdamW lr 1e-3 cosine, AMP, batch 32–64.
- **Generic `Instrument` model**: tracks are agnostic, labeled via rules (`labeling/rules.py`)
  or optional LLM (`LYRE_LLM_ENDPOINT`/`LYRE_LLM_KEY` env vars), consumed generically downstream; supports multi-guitar.
- **Alternative tunings**: presets in `GUITAR_TUNINGS`/`BASS_TUNINGS` (standard, half-step-down, drop-d/c/b/a; custom pitch lists via `resolve_tuning`).
- **Time signatures first-class** (any meter, e.g. 7/8) threaded through all outputs.
- **Output bundle**: `score.mid`, per-instrument `.mid`, `guitar.tab`, `bass.tab`, `drums.txt`,
  `score.gp5`, `score.musicxml`, `score.pdf` (verovio, Musescore-CLI fallback).

## Environment
- Python **3.14**, virtualenv at `.venv/`. Host runs CPU torch; training **requires GPU**.
- Key deps: torch 2.13+cu130, torchaudio 2.11, torchcodec 0.15 (required for torchaudio save),
  PyGuitarPro **0.11**, demucs, mir_eval, pretty_midi.
- Run python as `.venv/bin/python`, pytest as `.venv/bin/python -m pytest tests -q`.

## Repo layout (all under `lyre/`)
```
lyre/io/decode.py            audio decode (torchcodec/torchaudio)
lyre/separator/demucs.py     frozen Demucs stem separation
lyre/transcriber/            features, model, targets, dataset, index, metrics
lyre/tracking/               hmm.py (note tracking), midi_io.py (MIDI write, merge_instruments)
lyre/instruments.py          Instrument dataclass, split_guitar, FAMILIES, program_for
lyre/labeling/               rules.py, llm.py
lyre/arranger/arrange.py     resolve_tuning, _bar_boundaries, build_arrangement, GUITAR/BASS_TUNINGS
lyre/arranger/render/        ascii.py, musicxml.py, gp5.py, pdf.py
lyre/pipeline.py             Converter orchestrating convert/train
lyre/cli/main.py             CLI commands
lyre/scripts/                train.py, evaluate.py, prepare_data.py, export.py
configs/default.yaml, configs/guitar_finetune.yaml
tests/test_features.py       (4 passing)
tests/test_gp5.py            (5 passing)  GP5 round-trip via guitarpro.parse
```
Note: modules were initially written at repo root, then moved into `lyre/` — confirmed via import check.

## Completed & verified
- Design spec committed (`1114d04`).
- All modules implemented; scaffold uncommitted (`?` in git) — full tree under `lyre/`, `configs/`, `tests/`, `pyproject.toml` not yet committed.
- End-to-end smoke (mock separator) produced score.mid, score.gp5, score.musicxml, guitar.tab, bass.tab, drums.txt; musicxml parses as valid XML.
- `tests/test_features.py`: **4 passing** (frame_rate 100.0, feature shape, MultiPitchNet output `(2,128,128)`/`(2,128,1)`, param count <30M).

## Gotchas already fixed (save the next agent time)
- Model forward: `(batch, n_notes, time)` via `freq-mean` over bins — `.squeeze(3).transpose(1,2)`, **not** einsum (ops were mis-ordered).
- Features: `F.spectrogram` + `F.melscale_fbanks` + `torch.matmul` (einsum ops mis-ordered).
- `frames_to_notes` honors only `min_note_sec`/`merge_gap_sec` kwargs.
- `pretty_midi.TimeSignature` requires `time`; `PrettyMIDI()` takes no `time_signature_changes` kwarg → set attr post-init.
- `Track(song=song)` auto-registers → must set `song.tracks = built` explicitly afterwards.
- `Voice(measure=measure)`, `Note(beat=beat)` required (auto-register).
- `resolve_tuning(presets=...)` must be passed per-instrument preset registry (it is not global).

## RESOLVED — GP5 writer round-trip (was the blocker)
**PyGuitarPro 0.11 is NOT broken.** A canonical minimal song round-trips fine; the earlier
"library bug" conclusion was wrong (the minimal repro had itself been built with the same
bad assumptions). Four real bugs in `lyre/arranger/render/gp5.py`, all fixed:
1. `measure.voices = [voice]` dropped the mandatory 2nd voice. GP5 always reads
   `Measure.maxVoices == 2`, so the reader desynced → the "voice 2 / beat 3" errors.
   Fix: reuse `measure.voices[0]` and clear its beats; never replace the list.
2. `GuitarString(number=i, ...)` was 0-based and tunings were passed low-to-high. GP numbers
   strings **1..n from the highest-pitched string down**; the reader tests `1 << (7 - string.number)`,
   so string 0 set bit 7 and corrupted the note mask. Fix: `number=i+1` over `reversed(tuning)`.
3. `note.string = n_strings - 1 - n.string` was 0-based → same mask problem. Fix: `n_strings - n.string`.
4. `Song()` ships a default track **and header**, and `Track(song=...)` seeds one `Measure` per
   existing header — so every track carried a phantom extra measure and `len(measureHeaders)`
   drifted from the per-track measure count. Fix: `song.measureHeaders = []` **before** building tracks.

Empty arrangements now emit one placeholder guitar track instead of a 0-track song
(`writeSong` dereferences `song.tracks[0]` and would crash).

Verified: `tests/test_gp5.py` — 5 tests covering 4/4, 7/8, drop-D, exact pitch round-trip
(`string.value + fret` reconstructs the source pitches), header/measure-count agreement, and the
empty arrangement. `guitarpro.parse()` is the success gate. Full suite: **9 passed**.

<details><summary>Historical debug notes (kept for reference)</summary>
`lyre/arranger/render/gp5.py::write_gp5` produces a `.gp5` that **PyGuitarPro's own reader cannot
re-read**. Includes `_split_units`, `_bar_units`, `_note_beat`, `_second_to_units`.

### Key finding (isolated this session)
**This is a library bug, not just our code.** A canonical minimal PyGuitarPro song —
1 track (6 strings) + 1 measure + 1 beat — written with `guitarpro.write(..., version=(5,1,0))`
**fails to round-trip** through `guitarpro.parse()` in installed **pyguitarpro 0.11**:
```
GPException: reading track 1, measure 1, voice 2, beat 3, got error: unpack requires a buffer of 1 bytes
```
An even simpler empty-measure file fails at `reading track 1, measure 2, voice 2` and, with a note+rest,
at `beat 2, ValueERror: count must be less than or equal to 255` (inside `readIntByteSizeString`).

### Debug evidence
- `writeMeasure`/`writeVoice` emit ONLY `measure.voices[0]` as `I32(len(beats))` + beats (gp3.py).
- `writeBeat`: flags U8; status!=normal → `flags|=0x40` + status byte (`BeatStatus`: empty=0, normal=1, rest=2);
  duration as `I8(bit_length(value)-3)` (quarter Duration(4) → 0); notes as U8 string-mask `1<<(7-string)` then per-note.
- `writeNote`: flags always `|0x20` (fret); `|0x01` when duration&tuplet set (default); writes type U8, duration I8,
  tuplet I8, velocity I8 if != `Velocities.default`, fret I8, effects. **Note read is symmetric** — note bytes match.
- Tracing `readByteSizeString` with `versionTuple=(5,1,0)`: reads header/track strings fine, then a garbage
  `count=147455` at the beat region → stream is misaligned *before* measure beats.
- With `versionTuple=None` the same file reads past **all** strings then dies with
  `TypeError: '>' not supported between instances of 'NoneType' and 'tuple'` (a `versionTuple` guard) — i.e. the
  byte consumption diverges between the version-detect path and the (5,1,0) path. **The reader overruns into
  "voice 2/measure 2", so the writer produces FEWER consumer bytes than one of the reader paths expects.**
- Writer `writeSong` field order (gp3 base): version, info, tripletFeel bool (reads `song.tracks[0].measures[0].tripletFeel`),
  tempo I32, key I32, `writeMidiChannels(tracks)`, `I32(len(tracks[0].measures))`, `I32(len(tracks))`,
  `writeMeasureHeaders`, `writeTracks`, `writeMeasures` (`zip(*partwiseMeasures)`). This is the prime suspect region:
  note `writeSong` accesses `song.tracks[0].measures[0]` even for 0-measure files (would crash) and the channel block
  is where voice/measure counts start drifting.

### Recommended next steps (pick one)
1. **Verify reality, stop trusting this reader**: open a `write_gp5`/minimal output in a real GP5 app or
   `tuxguitar`. The writer may be fine and only PyGuitarPro 0.11's reader is broken → then just ship files and
   validate externally.
2. **Upgrade/patch pyguitarpro**: try a newer version or monkeypatch `writeSong`/`readSong` so
   `writeMidiChannels`/measure-count blocks agree; pin the version that self-round-trips.
3. **Different writer**: if PyGuitarPro is unusable, replace `gp5.py` with a known-good GP5 encoder (e.g. write via
   `mingus`/`musescore` conversion, or emit GP3 format which the lib may round-trip).
4. Keep MusicXML + ASCII as the guaranteed-correct score outputs (both already verified) and treat GP5 as best-effort
   until (1)–(3) resolves.

Do **not** burn time adjusting `_split_units`/`_bar_units` numerics — the failure reproduces with a single beat and
0 remaining, including the empty issue is structural, not a beat-count math problem.
</details>

## Config defaults (configs/default.yaml)
- features: n_mels 229, n_fft 2048, f_min 30, f_max 16000, window_frames 128, hop_ms 10
- model: channels [32,64,96,128], n_notes 128
- train: batch 32, epochs 60, lr 1e-3, wd 1e-4, warmup 2, loss_onset_weight 0.5, amp, eval_every 2
- tracking: p_onset 0.05, p_sustain 0.95, min_note_sec 0.06, merge_gap_sec 0.03, min_velocity 0.2
- arrange: tempo 120, ts [4,4], guitar/bass tuning standard
- labeling: use_llm false; audio: sample_rate 44100, decode_channels 2

## Next actions (ordered)
1. Broaden test coverage beyond features + gp5: `arrange.voice_chord`/`_bar_boundaries`,
   `tracking/hmm.frames_to_notes`, `labeling/rules`, `musicxml`/`ascii` renderers.
2. Real end-to-end run with actual Demucs separation (so far only mock-separator smokes).
3. Training run — needs GPU; host is CPU-only torch. Data prep via `lyre/scripts/prepare_data.py`.
4. `write_pdf` (verovio / MuseScore-CLI fallback) is the one bundle format never exercised — verify it.

Checked and deliberately NOT changed: `lyre/arranger/__init__.py` and `render/__init__.py`
re-exports all resolve and `pipeline.py` imports through them — they are a live public API, not dead code.