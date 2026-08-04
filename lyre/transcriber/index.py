import json


def load_index(path):
    with open(path) as fh:
        return json.load(fh)["entries"]


def save_index(path, entries):
    with open(path, "w") as fh:
        json.dump({"entries": entries}, fh, indent=2)


def iter_entries(entries, source_weights=None):
    if source_weights is None:
        for entry in entries:
            yield entry
        return
    import random

    rng = random.Random(0)
    buckets = {}
    for entry in entries:
        buckets.setdefault(entry["source"], []).append(entry)
    present = {name for name in buckets}
    weights = {name: source_weights.get(name, 1.0) for name in present}
    total = sum(weights.values())
    weights = {name: w / total for name, w in weights.items()}
    while True:
        source = rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
        yield rng.choice(buckets[source])
