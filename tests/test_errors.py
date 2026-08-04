"""The exception hierarchy is part of the public interface.

Callers -- the CLI, the pipeline's per-stage handlers, and code outside the
package -- catch by base class. Which base a concrete error derives from, and
which name a module re-exports, therefore decide what a caller catches. Both are
pinned here because both have already drifted once: ``pdf.py`` defined its own
``PdfUnavailable``, so catching ``lyre.errors.PdfUnavailable`` did not catch
what ``write_pdf`` raised.
"""

import inspect

import pytest

from lyre import errors
from lyre.errors import (
    AudioDecodeError,
    CheckpointNotFound,
    ConfigError,
    LabelingError,
    LyreError,
    PdfBackendMissing,
    PdfUnavailable,
)


def _exported_classes():
    return [getattr(errors, name) for name in errors.__all__]


def test_all_lists_only_names_the_module_defines():
    for name in errors.__all__:
        assert hasattr(errors, name), "errors.__all__ names a missing %s" % name
        assert inspect.isclass(getattr(errors, name))


def test_every_exported_error_is_a_lyre_error():
    for cls in _exported_classes():
        assert issubclass(cls, LyreError), "%s is not a LyreError" % cls.__name__


def test_every_error_class_in_the_module_is_exported():
    """An error not in ``__all__`` is one nothing documents how to catch."""
    defined = [
        obj
        for obj in vars(errors).values()
        if inspect.isclass(obj)
        and issubclass(obj, BaseException)
        and obj.__module__ == errors.__name__
    ]
    assert sorted(c.__name__ for c in defined) == sorted(errors.__all__)


def test_lyre_error_derives_from_runtime_error():
    """Deliberate, not incidental.

    ``AudioDecodeError`` and ``PdfUnavailable`` were plain ``RuntimeError``
    subclasses before this hierarchy existed, and callers outside the package
    still wrap ``load_audio`` / ``write_pdf`` in ``except RuntimeError``.
    Narrowing the base to ``Exception`` would silently stop those from catching.
    """
    assert issubclass(LyreError, RuntimeError)
    assert issubclass(LyreError, Exception)


@pytest.mark.parametrize(
    "cls",
    [AudioDecodeError, ConfigError, PdfUnavailable, PdfBackendMissing,
     CheckpointNotFound, LabelingError],
)
def test_a_concrete_error_is_catchable_as_runtime_error(cls):
    with pytest.raises(RuntimeError):
        raise cls("boom")
    with pytest.raises(LyreError):
        raise cls("boom")


def test_pdf_backend_missing_is_a_kind_of_pdf_unavailable():
    """A machine with no MuseScore is a different problem from one that crashed.

    ``PdfBackendMissing`` says "no binary on PATH": every other output was
    produced correctly and the caller should report and carry on. A MuseScore
    crash or timeout stays plain ``PdfUnavailable`` -- that is a failed run. A
    caller that does not care about the distinction must still catch both.
    """
    assert issubclass(PdfBackendMissing, PdfUnavailable)
    assert PdfBackendMissing is not PdfUnavailable
    with pytest.raises(PdfUnavailable):
        raise PdfBackendMissing("no mscore on PATH")


def test_the_pdf_distinction_is_one_way():
    """A render failure must not be mistaken for a missing backend."""
    with pytest.raises(PdfUnavailable) as exc:
        raise PdfUnavailable("musescore exited 1")
    assert not isinstance(exc.value, PdfBackendMissing)


def test_legacy_import_paths_are_the_same_objects():
    """The re-exports are identities, not lookalikes.

    Two classes with the same name is exactly the bug: ``except
    lyre.errors.PdfUnavailable`` silently stops catching what ``write_pdf``
    raises, and nothing about the code reads wrong.
    """
    from lyre.arranger.render.pdf import PdfUnavailable as pdf_error
    from lyre.io.decode import AudioDecodeError as decode_error

    assert decode_error is AudioDecodeError
    assert pdf_error is PdfUnavailable


def test_the_error_classes_carry_their_message():
    for cls in _exported_classes():
        assert str(cls("the message")) == "the message"
