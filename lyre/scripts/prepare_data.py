import hashlib
import os

import yaml

from lyre.transcriber.index import save_index

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".mp4", ".ogg", ".aiff"}
MIDI_EXTS = {".mid", ".midi"}


def discover_slakh(root):
    entries = []
    root = os.path.expanduser(root)
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
            entries.append({"audio": audio, "midi": midi, "source": "slakh", "instrument": stem_id})
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
    by_stem = {}
    for f in os.listdir(path):
        stem, ext = os.path.splitext(f)
        by_stem.setdefault(stem, {})[ext] = os.path.join(path, f)
    for stem, files in by_stem.items():
        audio = next((p for ext, p in files.items() if ext.lower() in AUDIO_EXTS), None)
        midi = next((p for ext, p in files.items() if ext.lower() in MIDI_EXTS), None)
        if audio and midi:
            entries.append({"audio": audio, "midi": midi, "source": "private", "instrument": "unknown"})
    return entries


def _split(entries):
    buckets = {"train": [], "val": [], "test": []}
    for e in entries:
        key = e["audio"]
        h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100
        bucket = "train" if h < 80 else ("val" if h < 90 else "test")
        buckets[bucket].append(e)
    return buckets


def main(config_path):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    sources = config.get("data", {}).get("sources", {})
    all_entries = []
    for name, path in sources.items():
        fn = {"slakh": discover_slakh, "maestro": discover_maestro, "guitar": discover_guitarset}.get(name)
        if name == "private":
            paths = path if isinstance(path, list) else [path]
            for p in paths:
                entries = discover_private(p)
                print(f"{name}: {len(entries)}")
                all_entries.extend(entries)
            continue
        if not fn:
            print(f"unknown source {name!r}, skipping")
            continue
        entries = fn(path)
        print(f"{name}: {len(entries)}")
        all_entries.extend(entries)

    index_dir = config.get("data", {}).get("index_dir", "data/index")
    os.makedirs(index_dir, exist_ok=True)
    buckets = _split(all_entries)
    for name, entries in buckets.items():
        save_index(os.path.join(index_dir, f"{name}.json"), entries)
        print(f"wrote {name}.json ({len(entries)})")
    by_source = {}
    for e in all_entries:
        by_source.setdefault(e["source"], []).append(e)
    for source, entries in by_source.items():
        sbuckets = _split(entries)
        save_index(os.path.join(index_dir, f"{source}.json"), entries)
        for name, sub in sbuckets.items():
            save_index(os.path.join(index_dir, f"{source}_{name}.json"), sub)