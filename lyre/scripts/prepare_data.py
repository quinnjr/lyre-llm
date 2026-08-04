import hashlib
import os

import yaml

from lyre.config import load_config
from lyre.errors import LyreError
from lyre.instruments import (
    FAMILY_BASS,
    FAMILY_DRUMS,
    FAMILY_GUITAR,
    FAMILY_KEYS,
    FAMILY_VOCALS,
    family_of_label,
    normalize_label,
)
from lyre.reporting import warn
from lyre.transcriber.index import save_index

BUCKETS = ("train", "val", "test")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".mp4", ".ogg", ".aiff"}
MIDI_EXTS = {".mid", ".midi"}

# The instrument vocabulary stamped on every index entry. It is a single shared
# constant because four separate consumers key on it -- the per-instrument eval
# table, the no-weak-voice gate, the fret-playability proxy, and the Demucs stem
# lookup behind --through-separator -- and a value none of them recognise is
# indistinguishable from an instrument that simply scored badly.
INSTRUMENTS = ("bass", "drums", "guitar", "piano", "vocals", "other")

# Instrument families collapse onto that vocabulary. Anything else (strings,
# synths, organs) is "other": the model still trains on it, but it is not a voice
# the arranger writes a part for.
_FAMILY_TO_INSTRUMENT = {
    FAMILY_GUITAR: "guitar",
    FAMILY_BASS: "bass",
    FAMILY_DRUMS: "drums",
    FAMILY_KEYS: "piano",
    FAMILY_VOCALS: "vocals",
}


def canonical_instrument(label):
    """Map any dataset's instrument string onto :data:`INSTRUMENTS`."""
    if not label:
        return "other"
    return _FAMILY_TO_INSTRUMENT.get(family_of_label(normalize_label(label)), "other")


def _slakh_stem_instruments(track_dir):
    """Read a Slakh track's ``metadata.yaml`` into ``{stem_id: instrument}``.

    Slakh names its stem files S00/S01/..., which say nothing about what plays on
    them; the instrument class lives in the track's metadata. A track without
    readable metadata yields an empty mapping and each stem falls back to
    ``"other"`` rather than aborting the scan of a 2100-track corpus.
    """
    path = os.path.join(track_dir, "metadata.yaml")
    try:
        with open(path) as fh:
            meta = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return {}
    stems = meta.get("stems")
    if not isinstance(stems, dict):
        return {}
    out = {}
    for stem_id, info in stems.items():
        if not isinstance(info, dict):
            continue
        label = info.get("inst_class") or info.get("midi_program_name") or info.get("program_num")
        out[str(stem_id)] = canonical_instrument(label)
    return out


def discover_slakh(root):
    entries = []
    root = os.path.expanduser(root)
    metadata_cache = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".flac"):
                continue
            stem_id = os.path.splitext(f)[0]
            audio = os.path.join(dirpath, f)
            midi = os.path.join(dirpath, "..", "MIDI", f"{stem_id}.mid")
            midi = os.path.normpath(midi)
            if not os.path.exists(midi):
                continue
            track_dir = os.path.normpath(os.path.join(dirpath, ".."))
            if track_dir not in metadata_cache:
                metadata_cache[track_dir] = _slakh_stem_instruments(track_dir)
            instrument = metadata_cache[track_dir].get(stem_id) or canonical_instrument(stem_id)
            # The raw stem id is kept alongside, not instead: it is the only way
            # back to the source file, but it is not an instrument name.
            entries.append({
                "audio": audio,
                "midi": midi,
                "source": "slakh",
                "instrument": instrument,
                "stem_id": stem_id,
            })
    return entries


def discover_maestro(root):
    root = os.path.expanduser(root)
    entries = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext == ".wav" and os.path.exists(os.path.join(dirpath, stem + ".midi")):
                entries.append({
                    "audio": os.path.join(dirpath, f),
                    "midi": os.path.join(dirpath, stem + ".midi"),
                    "source": "maestro",
                    "instrument": "piano",
                })
    return entries


def discover_guitarset(root):
    root = os.path.expanduser(root)
    entries = []
    audio_dir = os.path.join(root, "audio_midi")
    if os.path.isdir(audio_dir):
        for f in os.listdir(audio_dir):
            stem, ext = os.path.splitext(f)
            if ext == ".wav" and os.path.exists(os.path.join(audio_dir, stem + ".midi")):
                entries.append({
                    "audio": os.path.join(audio_dir, f),
                    "midi": os.path.join(audio_dir, stem + ".midi"),
                    "source": "guitar",
                    "instrument": "guitar",
                })
    return entries


def discover_private(path):
    path = os.path.expanduser(path)
    entries = []
    # A missing root is "this corpus is not here", the same as it is for the
    # other three discoverers. Raising instead means `private` -- the last key in
    # the shipped config, and a placeholder path -- aborts the scan before the
    # curated "check the configured paths" error can ever be reached.
    if not os.path.isdir(path):
        return entries
    by_stem = {}
    for f in os.listdir(path):
        stem, ext = os.path.splitext(f)
        by_stem.setdefault(stem, {})[ext] = os.path.join(path, f)
    for stem, files in by_stem.items():
        audio = next((p for ext, p in files.items() if ext.lower() in AUDIO_EXTS), None)
        midi = next((p for ext, p in files.items() if ext.lower() in MIDI_EXTS), None)
        if audio and midi:
            entries.append({"audio": audio, "midi": midi, "source": "private", "instrument": "other"})
    return entries


def _bucket_of(entry):
    """Deterministic train/val/test assignment, hashed on the audio path."""
    h = int(hashlib.sha1(entry["audio"].encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def _index_name(data, bucket):
    """Output filename for ``bucket``, honouring data.{train,val,test}_index.

    Reading the same key that train/evaluate read means renaming an index in the
    YAML cannot leave prepare-data writing one filename while training looks for
    another.
    """
    return data.get(f"{bucket}_index") or f"{bucket}.json"


def _per_source_names(sources):
    """Every filename the per-source pass will write, for collision detection."""
    names = set()
    for source in sources:
        names.add(f"{source}.json")
        for bucket in BUCKETS:
            names.add(f"{source}_{bucket}.json")
    return names


def main(config_path):
    config = load_config(config_path)
    failures = []
    notes = []
    data = config.get("data") or {}
    sources = data.get("sources") or {}
    if not sources:
        raise LyreError(
            f"{config_path}: data.sources is missing or empty, so there is nothing to "
            "index. Add a data.sources mapping of dataset name -> root path "
            "(slakh, maestro, guitar, private)."
        )

    # The per-source pass writes `{source}_{bucket}.json` and runs after the
    # global pass, so a bucket index configured to one of those names is
    # overwritten by a single corpus and the mixture silently disappears. Caught
    # before discovery walks anything.
    reserved = _per_source_names(sources)
    for bucket in BUCKETS:
        filename = _index_name(data, bucket)
        if filename in reserved:
            raise LyreError(
                f"{config_path}: data.{bucket}_index is {filename!r}, which is also "
                "the name prepare-data gives the per-source index for "
                f"{filename.split('_')[0].removesuffix('.json')!r}. The per-source "
                "pass runs second, so this run would write that filename twice and "
                "leave one corpus where the mixture should be. A config that trains "
                "on a per-source index gets its indexes from the pretraining "
                f"config: run prepare-data with that one instead, or rename "
                f"data.{bucket}_index here."
            )

    discoverers = {
        "slakh": discover_slakh,
        "maestro": discover_maestro,
        "guitar": discover_guitarset,
    }
    all_entries = []
    unknown = []
    for name, path in sources.items():
        # An explicit `null` is how you say "I have not pointed this at anything
        # yet". Caught here so the error names the key; letting it through would
        # surface as a bare TypeError inside os.path.expanduser.
        if path is None or (isinstance(path, list) and not [p for p in path if p]):
            raise LyreError(
                f"{config_path}: data.sources.{name} has no path. Set it to the "
                "dataset root, or remove the key if you do not have that corpus."
            )
        if name == "private":
            paths = path if isinstance(path, list) else [path]
        elif name in discoverers:
            paths = [path]
        else:
            # A typo'd source key is never intentional, and the mixture it
            # produces is not the one the config describes -- so this run failed,
            # even though the indexes it wrote are internally consistent.
            unknown.append(name)
            warn(
                f"unknown source {name!r} in data.sources, skipping; known sources "
                "are slakh, maestro, guitar, private",
                failures,
            )
            continue
        entries = []
        for p in paths:
            if not os.path.isdir(os.path.expanduser(p)):
                warn(f"data.sources.{name}: {p} is not a directory, nothing indexed", failures)
                continue
            entries.extend(discover_private(p) if name == "private" else discoverers[name](p))
        print(f"{name}: {len(entries)}")
        all_entries.extend(entries)

    if not all_entries:
        raise LyreError(
            "no audio/MIDI pairs were discovered under data.sources "
            f"({', '.join(sorted(sources))}); check the configured paths"
            + (f" (unknown sources: {', '.join(unknown)})" if unknown else "")
        )

    index_dir = data.get("index_dir", "data/index")
    os.makedirs(index_dir, exist_ok=True)

    # One pass: assign each entry its bucket once, and bucket per source at the
    # same time so no audio path is hashed twice.
    buckets = {name: [] for name in BUCKETS}
    per_source = {}
    for entry in all_entries:
        bucket = _bucket_of(entry)
        buckets[bucket].append(entry)
        per_source.setdefault(entry["source"], {n: [] for n in BUCKETS})[bucket].append(entry)

    written = []
    for bucket in BUCKETS:
        filename = _index_name(data, bucket)
        save_index(os.path.join(index_dir, filename), buckets[bucket])
        written.append(filename)
        print(f"wrote {filename} ({len(buckets[bucket])})")

    # Per-source indexes, split and combined. The combined `{source}.json` is
    # what you point an ad-hoc eval or a single-corpus experiment at; dropping it
    # broke those silently, because nothing errors on an index that is merely
    # absent until the path is opened.
    #
    # These names are deliberately NOT run through _index_name: that helper
    # resolves the three global bucket indexes (data.train_index and friends),
    # while these follow the fixed `{source}_{bucket}.json` convention the
    # fine-tune config already names explicitly (guitar_train.json, ...).
    for source, sbuckets in sorted(per_source.items()):
        combined = []
        for bucket in BUCKETS:
            filename = f"{source}_{bucket}.json"
            save_index(os.path.join(index_dir, filename), sbuckets[bucket])
            written.append(filename)
            print(f"wrote {filename} ({len(sbuckets[bucket])})")
            combined.extend(sbuckets[bucket])
        filename = f"{source}.json"
        save_index(os.path.join(index_dir, filename), combined)
        written.append(filename)
        print(f"wrote {filename} ({len(combined)})")
    return {
        "index_dir": index_dir,
        "entries": len(all_entries),
        "written": written,
        "failures": failures,
        "notes": notes,
    }
