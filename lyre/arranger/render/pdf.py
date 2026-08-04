# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import os
import shutil
import subprocess

# Re-exported, not redefined: a second class of the same name would not be
# caught by callers doing `except lyre.errors.PdfUnavailable`.
from lyre.errors import PdfBackendMissing, PdfUnavailable

__all__ = ["PdfBackendMissing", "PdfUnavailable", "write_pdf"]


MUSESCORE_BINARIES = ("musescore", "musescore4", "musescore3", "mscore")

# Everything MuseScore needs to find its libraries, config and fonts — and
# nothing else. Notably absent: LYRE_* (LYRE_LLM_KEY in particular).
ENV_ALLOWLIST = frozenset(
    (
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "DISPLAY",
        "LD_LIBRARY_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
    )
)


def write_pdf(musicxml_path, pdf_path):
    """Render MusicXML to PDF with the MuseScore CLI.

    Verovio is deliberately not used here: its Python toolkit renders SVG,
    MIDI and timemaps but has no PDF writer, so the previous `renderToPdf`
    call could never have succeeded.
    """
    musescore = None
    for name in MUSESCORE_BINARIES:
        musescore = shutil.which(name)
        if musescore:
            break
    if not musescore:
        # No binary at all is a property of the machine, not a failed render, so
        # it carries the subclass callers can treat as advisory. Everything below
        # this point means MuseScore ran and did not produce a PDF, which is.
        raise PdfBackendMissing(
            "PDF export needs a MuseScore CLI on PATH (tried: "
            + ", ".join(MUSESCORE_BINARIES)
            + ")"
        )
    # MuseScore is a Qt app and aborts without a display server unless it is
    # told to use the offscreen platform plugin. Only an allowlist of the parent
    # environment is passed through: a child process's environment is readable
    # at /proc/<pid>/environ and is inherited by anything it spawns, and this
    # process holds LYRE_LLM_KEY, which a third-party binary has no use for.
    env = {"QT_QPA_PLATFORM": "offscreen"}
    for key, value in os.environ.items():
        if key in ENV_ALLOWLIST or key.startswith("XDG_"):
            env[key] = value
    # Absolute paths so that a filename beginning with "-" cannot be parsed by
    # MuseScore as an option.
    musicxml_path = os.path.abspath(musicxml_path)
    pdf_path = os.path.abspath(pdf_path)
    try:
        # Bounded: MuseScore can wedge on malformed input or a display probe,
        # and capture_output blocks us on its pipes while it does. A hang here
        # is the one failure the caller's per-stage isolation cannot recover
        # from, so cap it rather than waiting forever.
        subprocess.run(
            [musescore, "-o", pdf_path, musicxml_path],
            check=True,
            capture_output=True,
            env=env,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        _remove_partial(pdf_path)
        raise PdfUnavailable(
            f"{musescore} timed out after {exc.timeout}s rendering {musicxml_path}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        # MuseScore can create the output file before failing.
        _remove_partial(pdf_path)
        stderr = (exc.stderr or b"")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        raise PdfUnavailable(
            f"{musescore} failed with exit code {exc.returncode}: {stderr.strip()}"
        ) from exc
    except OSError as exc:
        raise PdfUnavailable(f"could not run {musescore}: {exc}") from exc


def _remove_partial(pdf_path):
    """Drop a half-written PDF so no caller mistakes it for a good render."""
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
