# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

"""The one place lyre reports something the user should see but that is not a
crash.

Every advisory goes to stderr, never stdout: the scripts write metric tables and
JSON to stdout, and a stray warning line in the middle of those corrupts them for
anything downstream that parses them.

`sink` exists so a caller that also has to *return* what it warned about (the
converter, which reports per-stage failures and performance notes) records and
prints in one step, instead of building a list and printing it again later.

Messages are sanitised here rather than at each call site because some of them
quote text lyre did not write -- an HTTP reason phrase, a decoder's error
string, a filename. One escape sequence in any of those rewrites the terminal or
forges a preceding line in a CI log, and a rule that has to be remembered at
twenty call sites is a rule that will be missed at one.
"""

import sys


def sanitize(text):
    """Replace every non-printable character in ``text`` with ``?``.

    Carriage returns matter as much as ANSI escapes: ``\\r`` lets remote text
    overwrite the line already printed, including the prefix that says where the
    message came from.
    """
    return "".join(ch if ch.isprintable() or ch == " " else "?" for ch in str(text))


def warn(message, sink=None):
    """Emit an advisory or failure message to stderr, optionally recording it."""
    message = sanitize(message)
    sys.stderr.write("warning: %s\n" % message)
    if sink is not None:
        sink.append(message)


__all__ = ["warn", "sanitize"]
