import json


def load_index(path):
    with open(path) as fh:
        return json.load(fh)["entries"]


def save_index(path, entries):
    with open(path, "w") as fh:
        json.dump({"entries": entries}, fh, indent=2)
