"""Gates on the config: what it says, what reads it, and what it builds.

Three bug classes live here.

1. "Advertised option that does nothing." A key in ``configs/default.yaml``
   that no code ever reads is a lie in the documentation. The scan below proves
   every leaf key reaches real code, *in the section it is declared in*.

2. "Knob the code reads that the config never declares." The reverse direction:
   ``data.get("cache_bytes")`` is a real knob no config mentions, and a typo
   (``cache_byes``) is indistinguishable from it unless something checks.

3. "Config drifted away from the code it configures." ``model.channels`` is the
   dangerous one: a 0.49M-parameter schedule satisfies every assertion in the
   suite unless something builds a model *from the config*.
"""

import ast
import functools
import re
from pathlib import Path

import pytest
import yaml

from lyre.errors import ConfigError
from lyre.config import load_config
from lyre.tracking.hmm import N_PITCHES, TRACKING_KEYS
from lyre.transcriber.dataset import DEFAULT_AUGMENT
from lyre.transcriber.model import MultiPitchNet

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"
FINETUNE = ROOT / "configs" / "guitar_finetune.yaml"
SOURCE = ROOT / "lyre"

# Keys consumed by a lookup whose name is built at runtime, so no string
# literal for them exists anywhere in the source. Every entry names the module
# that consumes it; nothing goes here to silence a real miss.
#
# Whole mappings whose *keys* are user-chosen names.
DYNAMIC_CONTAINERS = {
    "data.sources": (
        "lyre/scripts/prepare_data.py iterates sources.items(); the corpus "
        "names are data, not identifiers"
    ),
    "data.mix_weights": (
        "lyre/scripts/train.py::_weighted_sampler does "
        "mix_weights.get(entry['source'])"
    ),
}

# Individual leaves looked up through a name list rather than written out.
DYNAMIC_ALLOWLIST = {
    key: "lyre.tracking.hmm.TRACKING_KEYS selects it for frames_to_notes"
    for key in TRACKING_KEYS
}
DYNAMIC_ALLOWLIST = {"tracking.%s" % k: v for k, v in DYNAMIC_ALLOWLIST.items()}

# Leaves the scanner sees read, but cannot attribute to their section because
# the code rebuilds the section as a fresh local mapping first. Matching these
# by leaf name alone is a deliberate, named weakening of the section check.
#
# `augment.*` is here because StemDataset merges the section over its defaults
# into a fresh mapping, so no `config["augment"]` subscript survives for the
# scanner to follow. The weakening is real but bounded: the section's key set is
# pinned exactly by test_the_augment_section_matches_the_dataset_defaults, which
# catches the renamed, missing or mis-sectioned key that leaf matching alone
# would wave through.
#
# These keys were previously matched by leaf for a WORSE reason: the read was
# `cfg[...]`, and `cfg` is a name the scanner reads as the config ROOT, so every
# augment key was recorded as a bogus top-level path. That local is now
# `augment_cfg`, which resolves to "a config mapping, section unknown" --
# recording the leaf and nothing false. test_no_augment_key_is_recorded_at_the
# _config_root keeps it that way.
PATH_UNRESOLVABLE = {
    "augment.%s" % key: (
        "lyre/transcriber/dataset.py::_augment_config merges the section over "
        "DEFAULT_AUGMENT into a fresh mapping, so the read is "
        "augment_cfg['%s'] with no 'augment' subscript in the chain" % key
    )
    for key in DEFAULT_AUGMENT
}

# The reverse direction: knobs the code reads that no shipped config declares.
# Every entry names where it is read and what happens without it. Anything read
# and not declared and not listed here is either a missing config key or a typo,
# and the two are indistinguishable from the code alone.
OPTIONAL_KEYS = {
    "data.cache_bytes": (
        "lyre/scripts/train.py passes it to StemDataset only when it is not "
        "None; otherwise dataset.DEFAULT_CACHE_BYTES (256 MB per worker) applies"
    ),
}


# --------------------------------------------------------------------------
# the config-key scan
# --------------------------------------------------------------------------

# A name that, on its own, identifies the WHOLE config mapping.
_ROOT_NAME = re.compile(r"^(config|cfg|conf|configuration)$")
# A name that identifies a config mapping, whole or sectional.
_SEED_NAME = re.compile(r"^(config|cfg|conf|configuration)$|_(cfg|config)$")
_FACTORIES = ("load_config",)
_GETTERS = ("get", "pop", "setdefault")

# Sentinel prefix: "this is a config mapping, but which section it is cannot be
# determined here". Reads through it yield a leaf name and no dotted path.
_UNRESOLVED = ("<unresolved>",)


class _ConfigReads(ast.NodeVisitor):
    """Collect the string keys read *out of a config mapping*, with their path.

    Deliberately narrower than a text search. ``config["n_mels"]`` and
    ``feats.get("n_mels")`` count; a comment, a docstring, the key of a
    ``DEFAULTS = {...}`` literal, and an unrelated ``n_mels = 229`` assignment
    do not. A regex accepts all four, which is how an entire dead ``augment:``
    block could pass: each key matches its own definition in
    ``DEFAULT_AUGMENT``.

    Every read is recorded twice: as a bare leaf name in ``keys``, and -- when
    the chain of subscripts back to the config root is resolvable -- as a dotted
    path in ``paths``. The dotted form is what makes a key's *section* part of
    its identity: ``config["features"]["n_mels"]`` records ``features.n_mels``,
    so declaring ``n_mels`` under ``model:`` instead is a miss rather than a
    coincidental match on the leaf name.
    """

    def __init__(self):
        # name -> (prefix, weak). A weak binding comes from a parameter name,
        # where the section is a guess; a strong one comes from an assignment
        # whose right-hand side we actually resolved, and overrules it.
        self.configish = {}
        self.keys = set()
        self.paths = set()
        self.changed = False

    def _mark(self, name, prefix, weak=False):
        known = self.configish.get(name)
        if known is None:
            self.configish[name] = (prefix, weak)
            self.changed = True
            return
        known_prefix, known_weak = known
        if known_weak and not weak:
            self.configish[name] = (prefix, False)
            self.changed = True
        elif known_weak == weak and known_prefix != prefix:
            if known_prefix is not _UNRESOLVED:
                self.configish[name] = (_UNRESOLVED, weak)
                self.changed = True

    def _prefix(self, node):
        """The dotted prefix ``node`` denotes, or ``False`` if it is not a config.

        Returns ``()`` for the config root, a tuple of section names for a
        resolved sub-mapping, and :data:`_UNRESOLVED` for a config mapping whose
        section cannot be determined.
        """
        if isinstance(node, ast.Name):
            if node.id in self.configish:
                return self.configish[node.id][0]
            if _SEED_NAME.search(node.id):
                return () if _ROOT_NAME.search(node.id) else _UNRESOLVED
            return False
        if isinstance(node, ast.Attribute):
            if _SEED_NAME.search(node.attr):
                return () if _ROOT_NAME.search(node.attr) else _UNRESOLVED
            return _UNRESOLVED if self._prefix(node.value) is not False else False
        if isinstance(node, ast.Subscript):
            inner = self._prefix(node.value)
            if inner is False:
                return False
            if inner is not _UNRESOLVED and _is_str_const(node.slice):
                return inner + (node.slice.value,)
            return _UNRESOLVED
        if isinstance(node, ast.BoolOp):  # config.get("x") or {}
            for value in node.values:
                prefix = self._prefix(value)
                if prefix is not False:
                    return prefix
            return False
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _GETTERS + ("copy",):
                inner = self._prefix(func.value)
                if inner is False:
                    return False
                if func.attr == "copy":
                    return inner
                if inner is not _UNRESOLVED and node.args and _is_str_const(node.args[0]):
                    return inner + (node.args[0].value,)
                return _UNRESOLVED
            if isinstance(func, ast.Name):
                if func.id in _FACTORIES:
                    return ()
                if func.id == "dict" and node.args:  # dict(config["tracking"])
                    return self._prefix(node.args[0])
        return False

    def _record(self, prefix, key):
        self.keys.add(key)
        if prefix is not _UNRESOLVED:
            self.paths.add(".".join(prefix + (key,)))

    def visit_arg(self, node):
        if _SEED_NAME.search(node.arg):
            root = () if _ROOT_NAME.search(node.arg) else _UNRESOLVED
            self._mark(node.arg, root, weak=True)

    def visit_Assign(self, node):
        self.generic_visit(node)
        prefix = self._prefix(node.value)
        if prefix is not False:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._mark(target.id, prefix)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        key = node.slice
        if _is_str_const(key):
            prefix = self._prefix(node.value)
            if prefix is not False:
                self._record(prefix, key.value)

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _GETTERS:
            if node.args and _is_str_const(node.args[0]):
                prefix = self._prefix(func.value)
                if prefix is not False:
                    self._record(prefix, node.args[0].value)

    def visit_Compare(self, node):
        self.generic_visit(node)
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)) and _is_str_const(node.left):
                prefix = self._prefix(comparator)
                if prefix is not False:
                    self._record(prefix, node.left.value)


def _is_str_const(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


@functools.lru_cache(maxsize=64)
def _scan(source):
    """``(leaf names, dotted paths)`` read out of a config mapping in ``source``."""
    tree = ast.parse(source)
    visitor = _ConfigReads()
    # Names are marked config-ish as they are seen, so a name used before its
    # binding is visited needs another pass. Converges in two or three.
    for _ in range(8):
        visitor.changed = False
        visitor.visit(tree)
        if not visitor.changed:
            break
    return frozenset(visitor.keys), frozenset(visitor.paths)


def _read_keys(source):
    return _scan(source)[0]


def _read_paths(source):
    return _scan(source)[1]


def _is_read(key, source):
    """True if ``key`` is read out of a config mapping somewhere in ``source``."""
    return key in _read_keys(source)


def _is_read_path(dotted, source):
    """True if ``dotted`` -- section included -- is read somewhere in ``source``."""
    return dotted in _read_paths(source)


def _leaf_keys(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_keys(value, path + (str(key),))
    else:
        yield path


def _all_paths(node, path=()):
    """Every dotted path in the config, interior nodes included."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield ".".join(path + (str(key),))
            yield from _all_paths(value, path + (str(key),))


@functools.lru_cache(maxsize=1)
def _source_files():
    return tuple(sorted(SOURCE.rglob("*.py")))


@functools.lru_cache(maxsize=1)
def _tree_reads():
    """``(keys, paths)`` for the package, scanned one file at a time.

    Per file and unioned, not one concatenated blob: a name is marked config-ish
    for the whole scan, so scanning everything together lets a plain ``data`` or
    ``model`` local in one module make every string-keyed subscript of that name
    in *every other* module count as a config read.
    """
    keys, paths = set(), set()
    for path in _source_files():
        file_keys, file_paths = _scan(path.read_text())
        keys |= file_keys
        paths |= file_paths
    return frozenset(keys), frozenset(paths)


@functools.lru_cache(maxsize=1)
def _source_text():
    return "\n".join(p.read_text() for p in _source_files())


def _unread_keys(config, paths=None, keys=None):
    """Config leaves that nothing under ``lyre/`` reads."""
    if paths is None or keys is None:
        keys, paths = _tree_reads()
    unread = []
    for path in _leaf_keys(config):
        dotted = ".".join(path)
        if dotted in DYNAMIC_ALLOWLIST:
            continue
        if any(dotted.startswith(c + ".") for c in DYNAMIC_CONTAINERS):
            continue
        if dotted in paths:
            continue
        if dotted in PATH_UNRESOLVABLE and path[-1] in keys:
            continue
        unread.append(dotted)
    return unread


def _undeclared_reads(config, paths):
    """Config reads in the code that the config never declares."""
    declared = set(_all_paths(config))
    leaves = {p[-1] for p in _leaf_keys(config)}
    undeclared = []
    for dotted in sorted(paths):
        if dotted in declared or dotted in OPTIONAL_KEYS:
            continue
        # A read whose section the scanner could not resolve carries no section
        # to check, so it matches on leaf name. A read whose section IS resolved
        # must match that section exactly.
        if "." not in dotted and dotted in leaves:
            continue
        undeclared.append(dotted)
    return undeclared


def _config():
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)


def _resolve(config, dotted):
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def test_config_file_parses():
    config = _config()
    assert isinstance(config, dict)
    assert config["features"]["n_mels"] == 229
    assert config["model"]["n_notes"] == 128


def test_the_key_scanner_bites():
    """Negative control. Without this the scan could be a no-op and look green."""
    keys, paths = _tree_reads()
    assert "n_mels" in keys
    assert "window_overlap" in keys
    assert "totally_unread_knob_xyz" not in keys

    # The section is part of a key's identity. n_mels is read as
    # config["features"]["n_mels"] in eight places and nowhere as
    # config["model"]["n_mels"], so only the first spelling counts as read.
    assert "features.n_mels" in paths
    assert "model.n_mels" not in paths
    assert "inference.window_overlap" in paths


def test_is_read_path_resolves_the_section_not_just_the_leaf():
    """Negative control for the section check itself.

    A leaf-name-only scan cannot tell a key in the right section from the same
    key in the wrong one, which is the failure mode that matters: the code does
    ``config["features"]["n_mels"]``, so a config declaring ``n_mels`` under
    ``model:`` is broken while looking perfectly well-read.
    """
    source = "def f(config):\n    feats = config['features']\n    return feats['n_mels']\n"
    assert _is_read("n_mels", source)
    assert _is_read_path("features.n_mels", source)
    assert not _is_read_path("model.n_mels", source)


@pytest.mark.parametrize(
    "snippet,dotted,expected",
    [
        ("def f(config):\n    return config['a']['b']\n", "a.b", True),
        ("def f(config):\n    return config['a']['b']\n", "c.b", False),
        ("def f(config):\n    return config.get('a', {}).get('b')\n", "a.b", True),
        ("def f(config):\n    d = dict(config['a'])\n    return d['b']\n", "a.b", True),
        ("def f(self):\n    return self.config['a']['b']\n", "a.b", True),
        # a section handed to a helper under a *_cfg name keeps its leaf but
        # loses its section, so nothing false is claimed about where it lives
        ("def f(a_cfg):\n    return a_cfg['b']\n", "a.b", False),
    ],
)
def test_is_read_path_discriminates(snippet, dotted, expected):
    assert _is_read_path(dotted, snippet) is expected


@pytest.mark.parametrize(
    "snippet,key,expected",
    [
        # prose and docstrings are not evidence
        ('# reads config["ghost"]\nx = 1\n', "ghost", False),
        ('"""Reads the ghost key from config."""\n', "ghost", False),
        # a defaults literal defining the key is not evidence of reading it
        ("DEFAULTS = {'ghost': 1}\n", "ghost", False),
        # nor is an unrelated local assignment
        ("def f():\n    ghost = 3\n    return ghost\n", "ghost", False),
        # nor a subscript of something that is not a config
        ("def f(row):\n    return row['ghost']\n", "ghost", False),
        # genuine reads
        ("def f(config):\n    return config['ghost']\n", "ghost", True),
        ("def f(config):\n    return config.get('ghost', 1)\n", "ghost", True),
        ("def f(config):\n    feats = config['features']\n    return feats['ghost']\n", "ghost", True),
        ("def f(train_cfg):\n    return train_cfg['ghost']\n", "ghost", True),
        ("def f(config):\n    t = dict(config['tracking'])\n    return t['ghost']\n", "ghost", True),
        ("def f(self):\n    return self.config['a']['ghost']\n", "ghost", True),
    ],
)
def test_is_read_discriminates(snippet, key, expected):
    assert _is_read(key, snippet) is expected


def test_every_leaf_key_is_read_somewhere():
    unread = _unread_keys(_config())
    assert unread == [], "config keys nothing under lyre/ reads: %s" % unread


def test_a_key_declared_in_the_wrong_section_is_reported_unread():
    """Negative control for the whole forward scan.

    Moving ``n_mels`` from ``features:`` to ``model:`` breaks eight call sites
    and changes nothing about which leaf names appear in the source, so a
    leaf-only scan stays green. The dotted scan must not.
    """
    config = _config()
    config["model"]["n_mels"] = config["features"].pop("n_mels")
    assert _unread_keys(config) == ["model.n_mels"]


def test_every_config_read_in_the_code_names_a_key_some_config_declares():
    """The reverse direction: code -> config.

    A read of a key no config declares is either a knob nobody can set or a
    typo, and nothing else in the suite can tell those apart from a working
    read. Genuine optional knobs go in OPTIONAL_KEYS with their default.
    """
    _, paths = _tree_reads()
    undeclared = _undeclared_reads(_config(), paths)
    assert undeclared == [], (
        "lyre/ reads config keys configs/default.yaml does not declare: %s. Add "
        "them to the config, fix the typo, or list them in OPTIONAL_KEYS."
        % undeclared
    )


def test_the_reverse_gate_catches_a_typo_in_an_optional_key():
    """Negative control for the reverse gate.

    ``data.get("cache_bytes")`` is allowlisted; ``data.get("cache_byes")`` reads
    identically at every other level and must not be.
    """
    typo = "def f(config):\n    return config['data'].get('cache_byes')\n"
    assert _undeclared_reads(_config(), _read_paths(typo)) == ["data.cache_byes"]

    real = "def f(config):\n    return config['data'].get('cache_bytes')\n"
    assert _undeclared_reads(_config(), _read_paths(real)) == []


def test_optional_keys_are_really_read_and_really_undeclared():
    """An OPTIONAL_KEYS entry that is now declared, or now unread, is stale."""
    _, paths = _tree_reads()
    declared = set(_all_paths(_config()))
    for dotted, reason in OPTIONAL_KEYS.items():
        assert dotted in paths, "OPTIONAL_KEYS names %s, which nothing reads" % dotted
        assert dotted not in declared, (
            "%s is declared in configs/default.yaml now; drop it from "
            "OPTIONAL_KEYS so the forward scan covers it" % dotted
        )
        assert "lyre/" in reason and "default" in reason.lower()


def test_path_unresolvable_entries_are_still_unresolvable_and_still_read():
    keys, paths = _tree_reads()
    dotted_config = {".".join(p) for p in _leaf_keys(_config())}
    for dotted, reason in PATH_UNRESOLVABLE.items():
        assert dotted in dotted_config, (
            "PATH_UNRESOLVABLE names %s, which the config no longer has" % dotted
        )
        assert dotted not in paths, (
            "%s resolves to its section now; drop the allowlist entry" % dotted
        )
        assert dotted.split(".")[-1] in keys
        assert "lyre/" in reason


def test_the_scanner_does_not_leak_config_ish_names_between_files():
    """Scanning one blob marks a name config-ish for every module at once.

    ``value``, ``src``, ``data``, ``model``, ``audio``, ``seed``, ``lr`` and
    ``tempo`` are all bound to something config-ish somewhere under ``lyre/``.
    Concatenating the package before parsing therefore lets an unrelated
    ``model["foo"]`` in any other module register as a config read. The per-file
    union is the honest answer; the blob agreeing with it is the proof that the
    channel is not currently being exploited.
    """
    blob_keys, blob_paths = _scan(_source_text())
    keys, paths = _tree_reads()
    leaked = sorted(blob_keys - keys)
    assert leaked == [], (
        "scanning lyre/ as one blob sees config reads the per-file scan does "
        "not: %s. Those keys are being credited to the wrong module." % leaked
    )
    assert blob_keys == keys
    # Paths only ever degrade in the blob: a name bound to two different
    # sections across modules collapses to "section unknown", so the blob can
    # lose a dotted path but must never invent one.
    invented = sorted(blob_paths - paths)
    assert invented == [], "blob scan invented config paths: %s" % invented


def test_allowlist_entries_still_exist_in_the_config():
    config = _config()
    dotted = {".".join(p) for p in _leaf_keys(config)}

    stale = sorted(set(DYNAMIC_ALLOWLIST) - dotted)
    assert stale == [], "allowlist references keys that are gone: %s" % stale

    for container in DYNAMIC_CONTAINERS:
        node = _resolve(config, container)
        assert isinstance(node, dict) and node, (
            "allowlisted container %s is missing or not a mapping" % container
        )

    # Every entry carries a reason naming its consumer.
    for reason in list(DYNAMIC_ALLOWLIST.values()) + list(DYNAMIC_CONTAINERS.values()):
        assert "lyre" in reason


def test_allowlisted_tracking_keys_are_really_consumed_dynamically():
    """The allowlist is only honest while TRACKING_KEYS actually selects them."""
    import inspect

    from lyre.tracking.hmm import frames_to_notes, tracking_params

    tracking = _config()["tracking"]
    assert set(DYNAMIC_ALLOWLIST) == {"tracking.%s" % k for k in tracking}

    selected = tracking_params(tracking)
    assert selected == tracking, "tracking_params dropped %s" % (
        sorted(set(tracking) - set(selected)),
    )
    params = inspect.signature(frames_to_notes).parameters
    for key in tracking:
        assert key in params


@pytest.mark.parametrize(
    "path,expected",
    [
        (("inference", "window_overlap"), 0.5),
        (("tracking", "p_onset"), 0.05),
        (("tracking", "p_sustain"), 0.95),
        (("tracking", "min_velocity"), 0.2),
        (("labeling", "use_llm"), False),
    ],
)
def test_contract_keys_live_where_the_contract_says(path, expected):
    """These keys are read by section, not by name; moving one breaks its caller."""
    node = _config()
    for key in path:
        assert key in node, "missing config key %s" % ".".join(path)
        node = node[key]
    assert node == expected


# --------------------------------------------------------------------------
# capacity: the config must build the model the spec asks for
# --------------------------------------------------------------------------


def test_default_config_builds_a_model_of_the_spec_capacity():
    """The config must build a model of the capacity the spec asks for.

    Asserting that ``channels`` is a sorted list of length >= 2 passes for
    [32, 64, 96, 128] -- a 0.49M-parameter model -- just as happily as for the
    real schedule. Only instantiating the model from the config catches that.
    """
    config = load_config(str(CONFIG))
    model = MultiPitchNet(
        n_mels=config["features"]["n_mels"],
        n_notes=config["model"]["n_notes"],
        channels=tuple(config["model"]["channels"]),
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert 15_000_000 <= n_params <= 30_000_000, (
        "configs/default.yaml builds a %d-parameter model; the spec target is "
        "15-30M (channels=%r)" % (n_params, config["model"]["channels"])
    )


def test_config_channels_match_the_model_default_schedule():
    """Config and code default must not drift apart silently."""
    import inspect

    default = inspect.signature(MultiPitchNet).parameters["channels"].default
    assert tuple(_config()["model"]["channels"]) == tuple(default)


def test_config_n_notes_matches_the_pitch_axis():
    assert _config()["model"]["n_notes"] == N_PITCHES


# --------------------------------------------------------------------------
# load_config: extends, deep merge, and the five failure modes
# --------------------------------------------------------------------------


def test_guitar_finetune_inherits_the_model_and_overrides_only_its_own_knobs():
    base = load_config(str(CONFIG))
    merged = load_config(str(FINETUNE))

    # `extends` itself is consumed, not passed through.
    assert "extends" not in merged

    # inherited verbatim -- a drifted channel schedule loads as an
    # incompatible state dict when the fine-tune resumes from a checkpoint.
    assert merged["model"] == base["model"]
    assert merged["model"]["channels"] == [128, 256, 512, 896]
    assert merged["augment"] == base["augment"]
    assert merged["features"]["min_note"] == base["features"]["min_note"] == 24
    assert merged["features"]["max_note"] == base["features"]["max_note"] == 95
    assert merged["features"]["n_mels"] == 229
    assert merged["tracking"] == base["tracking"]
    assert merged["data"]["index_dir"] == base["data"]["index_dir"]
    assert merged["data"]["num_workers"] == base["data"]["num_workers"]
    assert merged["data"]["sources"] == base["data"]["sources"]

    # The index files are inherited too: phase 2 trains on the same corpus and
    # becomes guitar-heavy purely through mix_weights. A guitar-only index would
    # contradict those weights (three of the four keys would match no entry and
    # fall back to 1.0) and would collide with the per-source index prepare-data
    # writes for the `guitar` source.
    assert merged["data"]["train_index"] == base["data"]["train_index"] == "train.json"
    assert merged["data"]["val_index"] == base["data"]["val_index"] == "val.json"
    assert merged["data"]["test_index"] == base["data"]["test_index"] == "test.json"

    # overridden
    assert merged["train"]["lr"] == 0.0002
    assert merged["train"]["epochs"] == 30
    assert merged["train"]["warmup_epochs"] == 1
    assert merged["train"]["eval_every"] == 1
    assert merged["run"]["out_dir"] == "runs/guitar_finetune"

    # untouched siblings of overridden keys survive the merge
    assert merged["train"]["batch_size"] == base["train"]["batch_size"]
    assert merged["train"]["weight_decay"] == base["train"]["weight_decay"]
    assert merged["run"]["seed"] == base["run"]["seed"]

    # mix_weights replaces wholesale; no stale key from the parent survives
    assert merged["data"]["mix_weights"] == {
        "slakh": 0.05,
        "guitar": 0.60,
        "maestro": 0.10,
        "private": 0.25,
    }
    assert "piano" not in merged["data"]["mix_weights"]
    assert sum(merged["data"]["mix_weights"].values()) == pytest.approx(1.0)

    # Every weight must name a corpus the inherited index actually contains, or
    # it is ignored and the corpus it meant to down-weight samples at 1.0.
    assert set(merged["data"]["mix_weights"]) == set(merged["data"]["sources"])


def test_guitar_finetune_builds_the_same_capacity_model_as_the_base():
    merged = load_config(str(FINETUNE))
    n_params = sum(
        p.numel()
        for p in MultiPitchNet(
            n_mels=merged["features"]["n_mels"],
            n_notes=merged["model"]["n_notes"],
            channels=tuple(merged["model"]["channels"]),
        ).parameters()
    )
    assert 15_000_000 <= n_params <= 30_000_000


def _write(path, text):
    path.write_text(text)
    return str(path)


def test_deep_merge_merges_mappings_key_by_key(tmp_path):
    _write(tmp_path / "base.yaml", "a:\n  x: 1\n  y: 2\nb: 3\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\na:\n  y: 20\n  z: 30\n")
    assert load_config(child) == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}


def test_deep_merge_replaces_lists_and_scalars_wholesale(tmp_path):
    _write(tmp_path / "base.yaml", "a:\n  ch: [1, 2, 3, 4]\n  s: 'old'\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\na:\n  ch: [9]\n  s: 'new'\n")
    # A half-inherited list ([9, 2, 3, 4]) is never what anyone means.
    assert load_config(child) == {"a": {"ch": [9], "s": "new"}}


def test_extends_resolves_relative_to_the_child_not_the_cwd(tmp_path):
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    _write(tmp_path / "sub" / "base.yaml", "a: 1\n")
    child = _write(nested / "child.yaml", "extends: ../base.yaml\nb: 2\n")
    assert load_config(child) == {"a": 1, "b": 2}


def test_extends_chains_grandparent_first(tmp_path):
    _write(tmp_path / "g.yaml", "a: 1\nb: 1\nc: 1\n")
    _write(tmp_path / "p.yaml", "extends: g.yaml\nb: 2\nc: 2\n")
    child = _write(tmp_path / "c.yaml", "extends: p.yaml\nc: 3\n")
    assert load_config(child) == {"a": 1, "b": 2, "c": 3}


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config(str(tmp_path / "nope.yaml"))
    assert "nope.yaml" in str(exc.value)


def test_missing_parent_names_the_parent(tmp_path):
    child = _write(tmp_path / "child.yaml", "extends: gone.yaml\na: 1\n")
    with pytest.raises(ConfigError) as exc:
        load_config(child)
    assert "gone.yaml" in str(exc.value)


def test_malformed_yaml_raises_config_error(tmp_path):
    path = _write(tmp_path / "bad.yaml", "a: [1, 2\nb: {\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "bad.yaml" in str(exc.value)
    assert "YAML" in str(exc.value)


@pytest.mark.parametrize("body", ["- 1\n- 2\n", "just a string\n", "42\n"])
def test_non_mapping_top_level_raises_config_error(tmp_path, body):
    path = _write(tmp_path / "top.yaml", body)
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "mapping" in str(exc.value)


def test_cyclic_extends_raises_config_error_instead_of_recursing(tmp_path):
    _write(tmp_path / "a.yaml", "extends: b.yaml\nx: 1\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\ny: 2\n")
    with pytest.raises(ConfigError) as exc:
        load_config(str(tmp_path / "a.yaml"))
    assert "cyclic" in str(exc.value)


def test_self_extends_raises_config_error(tmp_path):
    path = _write(tmp_path / "self.yaml", "extends: self.yaml\nx: 1\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "cyclic" in str(exc.value)


def test_n_notes_mismatch_raises_config_error(tmp_path):
    path = _write(tmp_path / "narrow.yaml", "model:\n  n_notes: 88\n  channels: [8]\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "88" in str(exc.value)
    assert str(N_PITCHES) in str(exc.value)


def test_n_notes_mismatch_is_caught_after_the_merge(tmp_path):
    """A child that narrows an inherited-good n_notes must still be rejected."""
    _write(tmp_path / "base.yaml", "model:\n  n_notes: 128\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\nmodel:\n  n_notes: 88\n")
    with pytest.raises(ConfigError):
        load_config(child)


def test_extends_must_be_a_string(tmp_path):
    path = _write(tmp_path / "bad.yaml", "extends: [a.yaml]\nx: 1\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "extends" in str(exc.value)


def test_empty_config_file_loads_as_an_empty_mapping(tmp_path):
    assert load_config(_write(tmp_path / "empty.yaml", "")) == {}


# --------------------------------------------------------------------------
# extends is the one place a config names another file to read
# --------------------------------------------------------------------------


def _tree(tmp_path):
    """A config directory whose containing tree is ``tmp_path``.

    ``extends`` may reach one level above the root config's own directory, so a
    config at ``<tmp>/configs/child.yaml`` may read ``<tmp>/**`` and nothing
    outside it.
    """
    configs = tmp_path / "configs"
    configs.mkdir()
    return configs


def test_extends_rejects_an_absolute_path(tmp_path):
    configs = _tree(tmp_path)
    outside = _write(tmp_path / "outside.yaml", "a: 1\n")
    child = _write(configs / "child.yaml", "extends: %s\nb: 2\n" % outside)
    with pytest.raises(ConfigError) as exc:
        load_config(child)
    assert "relative" in str(exc.value)


def test_extends_rejects_a_home_relative_path(tmp_path):
    configs = _tree(tmp_path)
    child = _write(configs / "child.yaml", "extends: ~/base.yaml\nb: 2\n")
    with pytest.raises(ConfigError) as exc:
        load_config(child)
    assert "relative" in str(exc.value)


@pytest.mark.parametrize("ref", ["base.txt", "base", "base.yaml.bak", "../../etc/shadow"])
def test_extends_rejects_a_non_yaml_suffix(tmp_path, ref):
    configs = _tree(tmp_path)
    child = _write(configs / "child.yaml", "extends: %s\nb: 2\n" % ref)
    with pytest.raises(ConfigError) as exc:
        load_config(child)
    assert "YAML" in str(exc.value)


def test_extends_rejects_a_parent_outside_the_config_tree(tmp_path):
    configs = _tree(tmp_path)
    # Two levels up from configs/ leaves the tree rooted at tmp_path.
    _write(tmp_path.parent / "escaped.yaml", "a: 1\n")
    child = _write(configs / "child.yaml", "extends: ../../escaped.yaml\nb: 2\n")
    with pytest.raises(ConfigError) as exc:
        load_config(child)
    assert "outside" in str(exc.value)


def test_extends_allows_a_sibling_and_a_parent_directory_inside_the_tree(tmp_path):
    """The rejections above must not have taken the legitimate layout with them."""
    configs = _tree(tmp_path)
    _write(tmp_path / "shared.yaml", "a: 1\n")
    _write(configs / "base.yaml", "b: 2\n")
    child = _write(configs / "child.yaml", "extends: base.yaml\nc: 3\n")
    assert load_config(child) == {"b": 2, "c": 3}
    other = _write(configs / "other.yaml", "extends: ../shared.yaml\nd: 4\n")
    assert load_config(other) == {"a": 1, "d": 4}


def test_extends_rejects_a_symlink_that_points_outside_the_tree(tmp_path):
    configs = _tree(tmp_path)
    target = tmp_path.parent / "linked_escape.yaml"
    target.write_text("a: 1\n")
    link = configs / "base.yaml"
    link.symlink_to(target)
    child = _write(configs / "child.yaml", "extends: base.yaml\nb: 2\n")
    try:
        with pytest.raises(ConfigError) as exc:
            load_config(child)
        assert "outside" in str(exc.value)
    finally:
        target.unlink()


# --------------------------------------------------------------------------
# required keys, encoding, and what an error message may quote
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,missing",
    [
        ("model:\n  n_notes: 128\n  channels: [8]\n", "features"),
        (
            "features:\n  n_mels: 229\n  n_fft: 2048\n  hop_ms: 10\n"
            "  window_frames: 128\n  min_note: 24\n  max_note: 95\n"
            "model:\n  n_notes: 128\n  channels: [8]\n"
            "audio:\n  sample_rate: 44100\n  decode_channels: 2\n",
            "tracking",
        ),
        (
            "features:\n  n_mels: 229\n  n_fft: 2048\n  hop_ms: 10\n"
            "  window_frames: 128\n  min_note: 24\n",
            "features.max_note",
        ),
        (
            "features:\n  n_mels: 229\n  n_fft: 2048\n  hop_ms: 10\n"
            "  window_frames: 128\n  min_note: 24\n  max_note: 95\n"
            "model:\n  n_notes: 128\n  channels: [8]\n"
            "audio:\n  sample_rate: 44100\n",
            "audio.decode_channels",
        ),
    ],
)
def test_a_run_config_missing_a_required_key_names_it(tmp_path, body, missing):
    path = _write(tmp_path / "run.yaml", body)
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "missing required key '%s'" % missing in str(exc.value)


def test_a_fragment_naming_no_run_section_is_not_required_to_be_complete(tmp_path):
    """`extends` is useless if every parent must be a whole run config."""
    path = _write(tmp_path / "fragment.yaml", "train:\n  lr: 0.001\nrun:\n  seed: 1\n")
    assert load_config(path) == {"train": {"lr": 0.001}, "run": {"seed": 1}}


def test_required_keys_are_checked_on_the_merged_config(tmp_path):
    """A child that supplies the missing half of a fragment is complete."""
    _write(
        tmp_path / "base.yaml",
        "features:\n  n_mels: 229\n  n_fft: 2048\n  hop_ms: 10\n"
        "  window_frames: 128\n  min_note: 24\n  max_note: 95\n"
        "tracking:\n  p_onset: 0.05\n",
    )
    child = _write(
        tmp_path / "child.yaml",
        "extends: base.yaml\nmodel:\n  n_notes: 128\n  channels: [8]\n"
        "audio:\n  sample_rate: 44100\n  decode_channels: 2\n",
    )
    merged = load_config(child)
    assert merged["features"]["n_mels"] == 229
    assert merged["model"]["channels"] == [8]


def test_a_non_utf8_config_is_named_as_such(tmp_path):
    path = tmp_path / "binary.yaml"
    path.write_bytes(b"a: \xff\xfe\x00not utf-8\n")
    with pytest.raises(ConfigError) as exc:
        load_config(str(path))
    assert "is not UTF-8 text" in str(exc.value)
    assert "binary.yaml" in str(exc.value)


def test_a_yaml_parse_error_never_quotes_the_file_it_read(tmp_path):
    """``extends`` names a file, so an error that echoes its contents is a leak.

    ``str()`` of a marked PyYAML error embeds the offending *source line*. With
    ``extends`` pointing at any readable path, interpolating that turns a config
    error into a read-a-line-of-any-file oracle.
    """
    secret = "SUPERSECRET ssh-rsa AAAA"
    path = _write(tmp_path / "leak.yaml", "k: %s: v\nb: 2\n" % secret)

    with pytest.raises(ConfigError) as exc:
        load_config(path)
    message = str(exc.value)

    # The fixture is only meaningful if PyYAML really would have leaked it.
    assert secret in str(exc.value.__cause__)

    assert "SUPERSECRET" not in message
    assert "ssh-rsa" not in message
    assert "AAAA" not in message
    # It still has to be actionable: the file, and where in it.
    assert "leak.yaml" in message
    assert "line 1" in message


def test_the_augment_section_matches_the_dataset_defaults():
    """The augment section and DEFAULT_AUGMENT must declare the same keys.

    The generic scanner can only match these by leaf name -- the dataset merges
    the section over its defaults into a fresh mapping, so no `augment`
    subscript survives for it to follow. Leaf matching would wave through a key
    moved to the wrong section, renamed on one side only, or dropped from the
    config while the code still reads it. Comparing the two key sets directly
    is both stronger and simpler: it is the invariant `_augment_config` already
    enforces at runtime, asserted at build time.
    """
    section = _config()["augment"]
    assert set(section) == set(DEFAULT_AUGMENT), (
        "configs/default.yaml augment section and "
        "lyre.transcriber.dataset.DEFAULT_AUGMENT disagree: "
        "only in config %s, only in code %s"
        % (sorted(set(section) - set(DEFAULT_AUGMENT)),
           sorted(set(DEFAULT_AUGMENT) - set(section)))
    )


def test_the_augment_section_check_would_catch_a_mis_sectioned_key():
    """Negative control: prove the assertion above can actually fail."""
    section = dict(_config()["augment"])
    moved = section.pop("gain_prob")
    assert moved is not None
    assert set(section) != set(DEFAULT_AUGMENT)


def test_no_augment_key_is_recorded_at_the_config_root():
    """A root-level `gain_prob` means an augment read was mis-attributed.

    `_item` once bound the merged mapping to a local called `cfg`, a name the
    scanner reads as the config root, so every augment key was recorded as a
    top-level path. Nothing declares those, so they were exempted rather than
    fixed.
    """
    paths = _read_paths(_source_text())
    for key in DEFAULT_AUGMENT:
        assert key not in paths, (
            "%r is recorded as a root-level config read; it belongs to the "
            "augment section" % key
        )
