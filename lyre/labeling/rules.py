# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

from lyre.errors import LabelingError
from lyre.instruments import Instrument, normalize_label, program_for
from lyre.reporting import warn as _report

_STEM_LABEL = {
    "bass": "bass",
    "drums": "drums",
    "guitar": "guitar",
    "piano": "piano",
    "vocals": "vocals",
    "other": "other",
}


def _split_guitar(instrument):
    from lyre.instruments import split_guitar

    rhythm, lead = split_guitar(instrument)
    if not rhythm or not lead:
        # The split did not apply, but this is still a guitar stem: rebuild it
        # so it gets the label and program every other stem gets. Returning the
        # instrument untouched left it named after whatever the caller passed
        # in, on program 0 (Acoustic Grand Piano).
        return [
            Instrument(
                name="guitar",
                notes=instrument.notes,
                program=program_for("guitar"),
                source=instrument.source,
            )
        ]
    out = []
    for notes, name in ((rhythm, "guitar rhythm"), (lead, "guitar lead")):
        out.append(
            Instrument(
                name=name,
                notes=notes,
                program=program_for(name),
                source=instrument.source,
            )
        )
    return out


def label_rules(instruments):
    labeled = []
    for instrument in instruments:
        stem = _STEM_LABEL.get(str(instrument.source).strip().lower())
        if stem:
            if stem == "guitar":
                labeled.extend(_split_guitar(instrument))
                continue
            name = stem
        else:
            name = normalize_label(instrument.name)
        labeled.append(
            Instrument(
                name=name,
                notes=instrument.notes,
                program=program_for(name),
                source=instrument.source,
            )
        )
    return labeled


def label_tracks(instruments, llm=None, sink=None, strict=False):
    """Label tracks with rules, optionally refined by an LLM.

    ``strict`` is set when the user explicitly asked for LLM labeling: they
    asked precisely because rule-based labels are not good enough, so a runtime
    LLM failure is raised rather than quietly downgraded. Otherwise the
    rule-based labels are returned and the failure is reported (never swallowed).

    Only :class:`~lyre.errors.LabelingError` is caught: a bug inside ``rename``
    must not be disguised as "the LLM was unreachable".
    """
    rules_out = label_rules(instruments)
    if llm is None:
        return rules_out
    try:
        return llm.rename(rules_out)
    except LabelingError as exc:
        if strict:
            raise
        _report(f"LLM labeling failed ({exc}); using rule-based labels", sink)
        return rules_out
