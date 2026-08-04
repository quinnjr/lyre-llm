from lyre.instruments import Instrument, normalize_label, program_for

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
        return [instrument]
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


def label_tracks(instruments, llm=None):
    rules_out = label_rules(instruments)
    if llm is None:
        return rules_out
    try:
        return llm.rename(rules_out)
    except Exception:
        return rules_out
