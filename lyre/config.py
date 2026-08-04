# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

"""Loading and validating run configs.

Every entry point (train, evaluate, export, prepare-data, the CLI) reads its YAML
through :func:`load_config` so they agree on four things:

* ``extends:`` — a config may name a parent to inherit from, so a fine-tune
  config carries only what it actually changes.  Duplicating the whole file
  instead means a knob edited in one copy and not the other (``model.channels``
  being the dangerous one) silently produces an incompatible checkpoint.
* Failures are reported against the file that caused them, by name.
* ``model.n_notes`` is checked against the pitch axis the rest of the system
  hardcodes.  Left unchecked, a mismatch trains and exports without complaint
  and only surfaces later inside the note tracker, nowhere near the config.
* The keys the rest of the system reads positionally are checked to exist.
  Everything downstream does ``config["model"]["channels"]``, so a config that
  omits a section fails with a bare ``KeyError`` from inside a constructor —
  no filename, no key name, and not an error type the CLI knows how to report.

``extends`` is also the one place a config file names another file to read, so
it is treated as untrusted input: relative, YAML-suffixed, and confined to the
config tree.  Nothing a config quotes back to the user may include the contents
of the file it read, which is why the PyYAML exception is never interpolated
whole — its ``str()`` embeds the offending source line.
"""

import os

import yaml

from lyre.errors import ConfigError
from lyre.tracking.hmm import N_PITCHES

EXTENDS_KEY = "extends"

YAML_SUFFIXES = (".yaml", ".yml")

# Keys read positionally downstream. A section with an empty tuple only has to
# be present; its contents are selected by name elsewhere (tracking) or are
# genuinely optional.
REQUIRED_KEYS = (
    ("features", ("n_mels", "n_fft", "hop_ms", "window_frames", "min_note", "max_note")),
    ("model", ("n_notes", "channels")),
    ("audio", ("sample_rate", "decode_channels")),
    ("tracking", ()),
)


def _yaml_detail(exc):
    """Describe a PyYAML failure without quoting the file it came from.

    ``str(exc)`` on a marked error includes the offending *source line*. Since
    ``extends`` lets one config name another file, interpolating that turns a
    config error into a read-any-file-you-can-name oracle: point it at a private
    key and the error message prints a line of it.
    """
    problem = getattr(exc, "problem", None)
    if not problem:
        return "could not be parsed"
    mark = getattr(exc, "problem_mark", None)
    if mark is None:
        return problem
    return "%s at line %d, column %d" % (problem, mark.line + 1, mark.column + 1)


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except UnicodeDecodeError as exc:
        # A UnicodeDecodeError is a ValueError, not an OSError, so it escapes
        # the handler below and reaches the user as a traceback with no
        # filename in it.
        raise ConfigError(f"{path}: is not UTF-8 text") from exc
    except OSError as exc:
        raise ConfigError(
            f"{path}: cannot read config file ({exc.strerror or exc})"
        ) from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: is not valid YAML ({_yaml_detail(exc)})") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"{path}: top level of a config must be a mapping, got "
            f"{type(loaded).__name__}"
        )
    return loaded


def _merge(base, override):
    """Deep-merge ``override`` onto ``base``, returning a new dict.

    Nested mappings merge key by key; anything else (scalars, and lists such as
    ``model.channels``) replaces wholesale, because a half-inherited list is
    never what someone means.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


def _config_tree(path):
    """The directory an ``extends`` chain starting at ``path`` may read from.

    One level above the root config's own directory, so the common layout —
    ``configs/experiments/tuned.yaml`` extending ``../default.yaml`` — keeps
    working while ``../../../etc/shadow`` does not.
    """
    return os.path.dirname(os.path.dirname(os.path.realpath(path)))


def _resolve_parent(path, parent_ref, tree):
    if not isinstance(parent_ref, str):
        raise ConfigError(
            f"{path}: '{EXTENDS_KEY}' must be a path string, got "
            f"{type(parent_ref).__name__}"
        )
    if not parent_ref:
        raise ConfigError(f"{path}: '{EXTENDS_KEY}' is empty")
    if os.path.isabs(parent_ref) or parent_ref.startswith("~"):
        raise ConfigError(
            f"{path}: '{EXTENDS_KEY}' must be relative to the config that "
            f"declares it, got {parent_ref!r}"
        )
    if not parent_ref.lower().endswith(YAML_SUFFIXES):
        raise ConfigError(
            f"{path}: '{EXTENDS_KEY}' must name a YAML file, got {parent_ref!r}"
        )
    # realpath, not abspath: otherwise a symlink inside the config directory is
    # a way around the containment check below.
    resolved = os.path.realpath(os.path.join(os.path.dirname(path), parent_ref))
    try:
        contained = os.path.commonpath([tree, resolved]) == tree
    except ValueError:  # different drives, so certainly not contained
        contained = False
    if not contained:
        raise ConfigError(
            f"{path}: '{EXTENDS_KEY}' {parent_ref!r} resolves outside the config "
            f"directory {tree}"
        )
    return resolved


def _load_chain(path, seen, tree):
    path = os.path.realpath(path)
    if path in seen:
        chain = " -> ".join(seen + [path])
        raise ConfigError(f"cyclic 'extends' in config files: {chain}")
    raw = _read(path)
    parent_ref = raw.pop(EXTENDS_KEY, None)
    if parent_ref is None:
        return raw
    parent_path = _resolve_parent(path, parent_ref, tree)
    parent = _load_chain(parent_path, seen + [path], tree)
    return _merge(parent, raw)


def _validate(config, path):
    model = config.get("model") or {}
    n_notes = model.get("n_notes")
    if n_notes is not None and n_notes != N_PITCHES:
        raise ConfigError(
            f"{path}: model.n_notes is {n_notes}, but the note tracker and the "
            f"frame targets are built for {N_PITCHES} pitches. A model of a "
            "different width trains and exports fine and then fails inside "
            "note tracking, so it is rejected here."
        )
    _require_keys(config, path)


def _require_keys(config, path):
    """Check the keys every consumer reads positionally.

    Skipped for a config that names none of the run sections at all: those are
    fragments (a parent holding one shared block, a test fixture), and demanding
    a whole run config from them would make ``extends`` useless. A config that
    names *any* run section is claiming to be one and must be complete.
    """
    if not any(section in config for section, _ in REQUIRED_KEYS):
        return
    for section, keys in REQUIRED_KEYS:
        present = config.get(section)
        if not isinstance(present, dict):
            if present is None or keys:
                raise ConfigError(f"{path}: missing required key '{section}'")
            continue
        for key in keys:
            if key not in present:
                raise ConfigError(
                    f"{path}: missing required key '{section}.{key}'"
                )


def load_config(path):
    """Read a YAML config, resolving ``extends``, and validate it."""
    config = _load_chain(path, [], _config_tree(path))
    _validate(config, path)
    return config


__all__ = ["load_config"]
