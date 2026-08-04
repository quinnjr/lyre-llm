"""Engraving MusicXML to PDF by shelling out to the MuseScore CLI.

MuseScore is a third-party GUI binary being driven headless. Three things about
that are load-bearing and none of them is visible in the output on a machine
where it happens to work: it must not inherit this process's secrets, it must
not be able to hang the run forever, and a render that failed must not leave a
file behind that later existence checks read as a good PDF.

Nothing here needs a real MuseScore: `shutil.which` and `subprocess.run` are
stubbed, so the tests run identically on a machine that has it and one that
does not.
"""

import os
import subprocess

import pytest

from lyre.arranger.render import pdf as pdf_mod
from lyre.arranger.render.pdf import MUSESCORE_BINARIES, write_pdf
from lyre.errors import PdfBackendMissing, PdfUnavailable

SECRET = "sk-live-abcdef0123456789"

FAKE_BINARY = "/usr/bin/musescore4"


def _no_binary(monkeypatch):
    monkeypatch.setattr(pdf_mod.shutil, "which", lambda name: None)


def _found(monkeypatch, path=FAKE_BINARY):
    monkeypatch.setattr(
        pdf_mod.shutil, "which", lambda name: path if name == "musescore" else None
    )


def _record_run(monkeypatch, effect=None):
    """Capture the subprocess.run call instead of making it."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if effect is not None:
            effect(argv, kwargs)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(pdf_mod.subprocess, "run", fake_run)
    return calls


def _paths(tmp_path):
    xml = tmp_path / "score.musicxml"
    xml.write_text("<score-partwise/>")
    return str(xml), str(tmp_path / "score.pdf")


# ------------------------------------------------------------- no backend


def test_no_musescore_on_path_raises_backend_missing(monkeypatch, tmp_path):
    _no_binary(monkeypatch)
    xml, pdf = _paths(tmp_path)

    with pytest.raises(PdfBackendMissing) as exc:
        write_pdf(xml, pdf)

    # The user has to be told what to install, so every name that was tried is
    # named: "musescore not found" is wrong advice on a machine with mscore.
    message = str(exc.value)
    for name in MUSESCORE_BINARIES:
        assert name in message
    # Advisory to callers that care about the distinction, still catchable by
    # the ones that do not.
    assert isinstance(exc.value, PdfUnavailable)


def test_every_candidate_binary_is_tried(monkeypatch, tmp_path):
    tried = []

    def which(name):
        tried.append(name)
        return None

    monkeypatch.setattr(pdf_mod.shutil, "which", which)
    xml, pdf = _paths(tmp_path)
    with pytest.raises(PdfBackendMissing):
        write_pdf(xml, pdf)
    assert tried == list(MUSESCORE_BINARIES)


def test_the_first_binary_found_stops_the_search(monkeypatch, tmp_path):
    tried = []

    def which(name):
        tried.append(name)
        return FAKE_BINARY if name == "musescore" else None

    monkeypatch.setattr(pdf_mod.shutil, "which", which)
    calls = _record_run(monkeypatch, effect=lambda argv, kw: open(argv[2], "wb").close())
    xml, pdf = _paths(tmp_path)
    write_pdf(xml, pdf)

    assert tried == ["musescore"]
    assert calls[0][0][0] == FAKE_BINARY


# ------------------------------------------------------------- environment


def test_the_child_environment_is_an_allowlist(monkeypatch, tmp_path):
    # A child process's environment is readable at /proc/<pid>/environ and is
    # inherited by anything it spawns. This process holds LYRE_LLM_KEY, which a
    # third-party engraver has no use for.
    monkeypatch.setenv("LYRE_LLM_KEY", SECRET)
    monkeypatch.setenv("LYRE_LLM_ENDPOINT", "https://api.example.com/v1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    _found(monkeypatch)
    calls = _record_run(monkeypatch)
    xml, pdf = _paths(tmp_path)
    write_pdf(xml, pdf)

    env = calls[0][1]["env"]
    assert SECRET not in repr(env)
    assert "LYRE_LLM_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    # ...and what MuseScore actually needs is still there.
    assert env["PATH"] == "/usr/bin:/bin"
    assert "PATH" in env


def test_qt_is_told_to_run_offscreen(monkeypatch, tmp_path):
    # MuseScore is a Qt app and aborts without a display server unless it is
    # told to use the offscreen platform plugin.
    _found(monkeypatch)
    calls = _record_run(monkeypatch, effect=lambda argv, kw: open(argv[2], "wb").close())
    xml, pdf = _paths(tmp_path)
    write_pdf(xml, pdf)

    assert calls[0][1]["env"]["QT_QPA_PLATFORM"] == "offscreen"


def test_xdg_variables_are_forwarded(monkeypatch, tmp_path):
    # MuseScore finds its config and fonts through these.
    monkeypatch.setenv("XDG_CONFIG_HOME", "/home/u/.config")
    monkeypatch.setenv("XDG_DATA_DIRS", "/usr/share")
    _found(monkeypatch)
    calls = _record_run(monkeypatch, effect=lambda argv, kw: open(argv[2], "wb").close())
    xml, pdf = _paths(tmp_path)
    write_pdf(xml, pdf)

    env = calls[0][1]["env"]
    assert env["XDG_CONFIG_HOME"] == "/home/u/.config"
    assert env["XDG_DATA_DIRS"] == "/usr/share"


# ------------------------------------------------------------- invocation


def test_the_render_is_bounded_by_a_timeout(monkeypatch, tmp_path):
    # MuseScore can wedge on malformed input or a display probe, and
    # capture_output blocks on its pipes while it does. A hang is the one
    # failure the caller's per-stage isolation cannot recover from.
    _found(monkeypatch)
    calls = _record_run(monkeypatch, effect=lambda argv, kw: open(argv[2], "wb").close())
    xml, pdf = _paths(tmp_path)
    write_pdf(xml, pdf)

    kwargs = calls[0][1]
    assert kwargs["timeout"] == 120
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True


def test_a_leading_dash_in_the_path_cannot_become_an_option(monkeypatch, tmp_path):
    _found(monkeypatch)
    calls = _record_run(monkeypatch, effect=lambda argv, kw: open(argv[2], "wb").close())

    xml = tmp_path / "-o.musicxml"
    xml.write_text("<score-partwise/>")
    pdf = tmp_path / "-score.pdf"
    monkeypatch.chdir(tmp_path)

    write_pdf("-o.musicxml", "-score.pdf")

    argv = calls[0][0]
    assert argv[-1].startswith("/")
    assert argv[-1] == str(xml)
    assert argv[2].startswith("/")
    assert argv[2] == str(pdf)


# ------------------------------------------------------------- failures


def test_a_timeout_removes_the_partial_pdf_and_is_not_advisory(monkeypatch, tmp_path):
    xml, pdf = _paths(tmp_path)

    def wedge(argv, kwargs):
        open(pdf, "wb").write(b"%PDF-1.4\ntrunc")
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    _found(monkeypatch)
    _record_run(monkeypatch, effect=wedge)

    with pytest.raises(PdfUnavailable) as exc:
        write_pdf(xml, pdf)

    assert "timed out" in str(exc.value)
    assert "120" in str(exc.value)
    # A half-written PDF is what every later existence check reads as a result.
    assert not os.path.exists(pdf)
    # MuseScore ran: this is a failed render, not a machine without the tool.
    assert not isinstance(exc.value, PdfBackendMissing)


def test_a_crash_removes_the_partial_pdf_and_reports_stderr(monkeypatch, tmp_path):
    xml, pdf = _paths(tmp_path)

    def crash(argv, kwargs):
        open(pdf, "wb").write(b"%PDF-1.4\ntrunc")
        raise subprocess.CalledProcessError(1, argv, stderr=b"boom")

    _found(monkeypatch)
    _record_run(monkeypatch, effect=crash)

    with pytest.raises(PdfUnavailable) as exc:
        write_pdf(xml, pdf)

    message = str(exc.value)
    assert "boom" in message
    assert "exit code 1" in message
    assert not os.path.exists(pdf)
    assert not isinstance(exc.value, PdfBackendMissing)


def test_a_crash_with_no_stderr_is_still_reported(monkeypatch, tmp_path):
    xml, pdf = _paths(tmp_path)

    def crash(argv, kwargs):
        raise subprocess.CalledProcessError(2, argv, stderr=None)

    _found(monkeypatch)
    _record_run(monkeypatch, effect=crash)

    with pytest.raises(PdfUnavailable) as exc:
        write_pdf(xml, pdf)
    assert "exit code 2" in str(exc.value)


def test_a_crash_with_text_stderr_is_still_reported(monkeypatch, tmp_path):
    xml, pdf = _paths(tmp_path)

    def crash(argv, kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="already a str")

    _found(monkeypatch)
    _record_run(monkeypatch, effect=crash)

    with pytest.raises(PdfUnavailable) as exc:
        write_pdf(xml, pdf)
    assert "already a str" in str(exc.value)


def test_an_unexecutable_binary_is_a_render_failure(monkeypatch, tmp_path):
    xml, pdf = _paths(tmp_path)

    def denied(argv, kwargs):
        raise PermissionError(13, "Permission denied")

    _found(monkeypatch)
    _record_run(monkeypatch, effect=denied)

    with pytest.raises(PdfUnavailable) as exc:
        write_pdf(xml, pdf)
    assert FAKE_BINARY in str(exc.value)
    # There *is* a binary; it just could not be run. Reporting that as "no
    # backend" would tell the user to install what they already have.
    assert not isinstance(exc.value, PdfBackendMissing)


def test_a_successful_render_leaves_the_pdf_alone(monkeypatch, tmp_path):
    _found(monkeypatch)
    _record_run(monkeypatch, effect=lambda argv, kw: open(argv[2], "wb").write(b"%PDF-1.4\n"))
    xml, pdf = _paths(tmp_path)

    write_pdf(xml, pdf)

    assert open(pdf, "rb").read() == b"%PDF-1.4\n"
