# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import json


def load_index(path):
    with open(path) as fh:
        return json.load(fh)["entries"]


def save_index(path, entries):
    with open(path, "w") as fh:
        json.dump({"entries": entries}, fh, indent=2)
