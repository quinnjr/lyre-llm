"""Project-wide exception hierarchy.

Everything lyre raises deliberately derives from :class:`LyreError` so callers
(and the CLI) can distinguish an expected, explainable failure from a genuine
bug.  Concrete errors are defined here and re-exported from the modules that
historically owned them, so existing import paths keep working.
"""


class LyreError(RuntimeError):
    """Base class for every error lyre raises on purpose.

    Deriving from :class:`RuntimeError` rather than :class:`Exception` is
    deliberate: ``AudioDecodeError`` and ``PdfUnavailable`` were plain
    ``RuntimeError`` subclasses before this hierarchy existed, and callers
    outside the package may still wrap ``load_audio`` / ``write_pdf`` in
    ``except RuntimeError``.  Keeping the wider base costs nothing and keeps
    those callers working.
    """


class AudioDecodeError(LyreError):
    """Raised when an input file cannot be decoded into usable audio."""


class PdfUnavailable(LyreError):
    """Raised when the PDF could not be produced.

    The base case is a *render* failure -- MuseScore ran and crashed, timed out,
    or could not be executed. That is a failure of the run.
    """


class PdfBackendMissing(PdfUnavailable):
    """Raised when no MuseScore CLI binary exists on PATH at all.

    Distinct from its base because it is a property of the machine, not of this
    run: every other output was produced correctly, so a caller should report it
    and carry on rather than failing. Callers that do not care about the
    distinction still catch it via ``PdfUnavailable``.
    """


class ConfigError(LyreError):
    """Raised when a run config is unreadable, malformed, or self-inconsistent."""


class CheckpointNotFound(LyreError):
    """Raised when an explicitly requested checkpoint path does not exist."""


class LabelingError(LyreError):
    """Raised when LLM-based track labeling was requested but cannot run."""


__all__ = [
    "LyreError",
    "AudioDecodeError",
    "ConfigError",
    "PdfUnavailable",
    "PdfBackendMissing",
    "CheckpointNotFound",
    "LabelingError",
]
