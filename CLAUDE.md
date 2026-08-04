# Working in this repo

Notes for Claude Code (and anyone else) working on lyre. Conventions and
hard-won invariants, not a tour — the README covers what the project is.

## Commands

```bash
.venv/bin/python -m pytest tests -q        # 630 tests, ~4s
.venv/bin/python -c "..."                  # never bare `python`
PYTHONPATH=. .venv/bin/python -m lyre.cli.main --help
```

The venv is Python 3.14 with torch 2.13+cu130, torchaudio 2.11, PyGuitarPro
0.11. `import lyre` needs `PYTHONPATH=.` unless installed with `-e`.

## House style

No type annotations. Plain functions, module-level constants in CAPS. Comments
explain **why**, never what — and never reference a review, a finding ID, an
audit, or a file outside the repo. A comment saying "fixes W-12" is meaningless
to the next reader; state the invariant instead.

## Invariants that will bite you

**Pitch arrays are `(n_frames, 128)`, everywhere.** `predict_track` returns it,
`frames_to_notes` asserts it. It was transposed once and the whole transcriber
silently emitted notes whose pitch was a frame index.

**`Note.velocity` is MIDI 0–127**, not 0–1. Writers clamp to 1–127; a 0 is a
silent note.

**GP5 durations use `UNITS_PER_WHOLE = 64`**, so a quarter is 16 units. Every
measure's beats must sum to `_bar_units(num, den)`. Also: a `Beat` left at the
attrs-default `BeatStatus.empty` makes PyGuitarPro's reader collapse every beat
in the measure into one, which looks like correct output until you check
durations.

**Drum times are quantised with `DRUM_TIME_PLACES`** in `render/_common.py`.
All three renderers must use it or a hit near a bar line lands in different
bars in different exports.

**Config keys are gated both directions.** `tests/test_config.py` parses `lyre/`
with an AST scanner and fails if a config key is read by nothing, or if the code
reads a key no config declares. Add a key to `configs/default.yaml` *and* read
it, or the suite fails. `guitar_finetune.yaml` uses `extends:` — change
`default.yaml` and it inherits.

**Two report channels, and they mean different things.** `failures` → the run
didn't do something it was asked to, exit 1. `notes` → advisory about the
material (silent stem, octave shift, no MuseScore), exit 0. Putting an advisory
on `failures` makes every real conversion exit 1; the inverse hides real
failures. `lyre/reporting.warn(message, sink=)` is the only emitter.

**Errors that reach the user subclass `LyreError`** (itself a `RuntimeError`).
Anything else escapes the CLI as a traceback with exit 1 instead of a clean
message with exit 2.

## Testing

Assert values, not shapes. This suite exists in its current form because an
earlier one passed while the model was 60× too small and every GP5 played at 4×
speed — both were "covered" by tests checking that outputs had the right shape
and weren't empty.

**Verify a fix by reverting it.** Break the thing you just fixed, confirm a test
fails, restore. A test that doesn't fail against the old behaviour isn't
testing the fix. Two tests have shipped here that were structurally incapable of
failing — one asserted a config value was "a sorted list of length ≥ 2", the
other iterated an empty list.

Back up with `cp` when experimenting. **Never `git checkout` a tracked file** —
staged and worktree state differ here and it has destroyed uncommitted work.

`tests/conftest.py` fails any test that opens a socket or downloads Demucs
weights, and restores torch RNG state between tests. Keep it that way.

## Not verified anywhere

Real Demucs separation, ONNX export (`onnx`/`onnxruntime` aren't installed), and
MuseScore PDF rendering. Their contracts are tested with fakes; the integrations
have never run. Don't claim otherwise in a commit message.

## Scope

Deliberately not implemented, and documented as such in the design spec: stem
re-mixing augmentation, streaming decode (separation and transcription are
chunked; decode reads the whole file), and Verovio PDF output (its Python
toolkit has no PDF writer). If you implement one, amend the spec too.
