# SPDX-License-Identifier: Elastic-2.0
# Copyright (c) 2026 Joseph R. Quinn

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from lyre.errors import LabelingError
from lyre.instruments import Instrument, LABEL_VOCAB, program_for, normalize_label
from lyre.reporting import sanitize

# Set to 1 to allow a plaintext http:// endpoint on the loopback interface (a
# local dev server, an ssh tunnel). Off by default because the API key travels
# in an Authorization header; a non-loopback host is refused even with the
# opt-out set, because "I am testing locally" never means "send my key across
# the network in cleartext".
ALLOW_INSECURE_ENV = "LYRE_LLM_ALLOW_INSECURE"

LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def _redact(endpoint):
    """Drop userinfo and query string before an endpoint reaches a log.

    ``https://user:sk-live-abc@host/v1`` is a perfectly valid endpoint and the
    credential in it would otherwise be copied into stderr and into the returned
    failure list, which on CI means into the build log.
    """
    try:
        parts = urllib.parse.urlsplit(endpoint)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "<unparseable endpoint>"
    if ":" in host:  # an IPv6 literal is only unambiguous inside brackets
        host = f"[{host}]"
    if port:
        host = f"{host}:{port}"
    if parts.username:
        host = f"<redacted>@{host}"
    return sanitize(urllib.parse.urlunsplit((parts.scheme, host, parts.path, "", "")))


def _origin(url):
    """The parts of a URL that decide whether two URLs are the same endpoint."""
    parts = urllib.parse.urlsplit(url)
    return (parts.scheme, parts.hostname, parts.port, parts.path)


def _check_endpoint(endpoint):
    parts = urllib.parse.urlsplit(endpoint)
    if parts.scheme == "https":
        return
    if parts.scheme in ("http", ""):
        if parts.scheme == "http" and os.environ.get(ALLOW_INSECURE_ENV):
            if parts.hostname in LOOPBACK_HOSTS:
                return
            raise LabelingError(
                f"LLM endpoint {_redact(endpoint)} is not https and is not on the "
                f"loopback interface; {ALLOW_INSECURE_ENV} only covers a local "
                "development server, because the API key would otherwise cross "
                "the network in cleartext"
            )
        raise LabelingError(
            f"LLM endpoint {_redact(endpoint)} is not https, so the API key would "
            f"be sent in cleartext; use https or set {ALLOW_INSECURE_ENV}=1 for a "
            "local development server"
        )
    # urllib happily opens file:// and ftp://, which changes what "request"
    # means; a scheme other than http(s) here is a typo, not an intention.
    raise LabelingError(
        f"LLM endpoint has unsupported scheme {parts.scheme!r}; expected https"
    )


def llm_from_env(required=False):
    """Build an :class:`LLMLabeler` from ``LYRE_LLM_*`` environment variables.

    When ``required`` is true (the user explicitly asked for LLM labeling) a
    missing ``LYRE_LLM_ENDPOINT`` is a hard :class:`~lyre.errors.LabelingError`
    rather than a silent fall back to rule-based labels.
    """
    endpoint = os.environ.get("LYRE_LLM_ENDPOINT")
    if not endpoint:
        if required:
            raise LabelingError(
                "LLM labeling was requested but LYRE_LLM_ENDPOINT is not set; "
                "set it (and optionally LYRE_LLM_KEY / LYRE_LLM_MODEL) or drop --llm"
            )
        return None
    return LLMLabeler(
        endpoint=endpoint,
        api_key=os.environ.get("LYRE_LLM_KEY", ""),
        model=os.environ.get("LYRE_LLM_MODEL", "gpt-4o-mini"),
    )


class LLMLabeler:
    def __init__(self, endpoint, api_key="", model="gpt-4o-mini", timeout=30):
        _check_endpoint(endpoint)
        self.endpoint = endpoint
        self.safe_endpoint = _redact(endpoint)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _prompt(self, instruments):
        rows = []
        for i, inst in enumerate(instruments):
            f = inst.features()
            rows.append(
                f"{i}: source={inst.source} name={inst.name!r} "
                f"notes={f['n_notes']} mean_polyphony={f['mean_polyphony']} "
                f"chordal_fraction={f['chordal_fraction']} single_line_fraction={f['single_line_fraction']} "
                f"pitch_range={f['pitch_min']}-{f['pitch_max']} duration_s={f['duration']}"
            )
        system = (
            "You label the instrument tracks of a multitrack transcription. "
            "Respond with JSON only: {\"tracks\":[{\"id\":<int>,\"label\":<str>}, ...]}. "
            f"Allowed labels: {LABEL_VOCAB}. A guitar source with both chordal and "
            "single-line passages should be split: emit two entries with the same id "
            "and labels 'guitar rhythm' and 'guitar lead'."
        )
        user = "\n".join(rows)
        return system, user

    def _request(self, instruments):
        system, user = self._prompt(instruments)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        # An empty key would otherwise be sent as the literal header
        # "Authorization: Bearer ", which some gateways reject differently from
        # no header at all, turning "you forgot the key" into an opaque 400.
        if self.api_key:
            # add_unredirected_header, not add_header: urllib's redirect handler
            # copies req.headers onto the request it sends to the redirect
            # target with no scheme and no host check, so a 302 from a
            # compromised or hostile endpoint is enough to walk the key out over
            # cleartext to anywhere. Unredirected headers are never copied.
            req.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        return req

    def _fetch(self, req):
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", "replace"), getattr(resp, "url", "")
        except urllib.error.HTTPError as exc:
            with exc:
                try:
                    detail = sanitize(exc.read().decode("utf-8", "replace")[:200])
                except Exception as read_exc:
                    detail = f"<could not read response body: {sanitize(read_exc)}>"
            # exc.reason is the server's own reason phrase and exc's str() can
            # carry raw bytes off the wire (BadStatusLine); neither is ours.
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} returned HTTP {exc.code} "
                f"{sanitize(exc.reason)}" + (f": {detail}" if detail else "")
            ) from exc
        except Exception as exc:
            raise LabelingError(
                f"LLM request to {self.safe_endpoint} failed: {sanitize(exc)}"
            ) from exc

    def _entry_instrument(self, entry, instruments):
        if not isinstance(entry, dict):
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} returned a track entry that is "
                f"not an object: {sanitize(repr(entry)[:80])}"
            )
        track_id = entry.get("id")
        # bool is an int; "id": true must not select track 1.
        if isinstance(track_id, bool) or not isinstance(track_id, int):
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} returned a track entry whose "
                f"'id' is not an integer: {sanitize(repr(track_id)[:40])}"
            )
        if not 0 <= track_id < len(instruments):
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} returned track id {track_id}, "
                f"but only {len(instruments)} tracks were sent"
            )
        label = entry.get("label")
        if label is not None and not isinstance(label, str):
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} returned a non-string label for "
                f"track {track_id}: {sanitize(repr(label)[:40])}"
            )
        base = instruments[track_id]
        name = normalize_label(label or base.name)
        return track_id, Instrument(
            name=name,
            notes=base.notes,
            program=program_for(name),
            source=base.source,
        )

    def rename(self, instruments):
        raw, final_url = self._fetch(self._request(instruments))
        if final_url and _origin(final_url) != _origin(self.endpoint):
            # The request body went to the redirect target, but the credential
            # did not (see _request) and this response is not the endpoint's.
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} redirected to "
                f"{_redact(final_url)}; a JSON completions POST has no reason to "
                "redirect, so the response is not trusted"
            )
        try:
            payload = json.loads(raw)
            content = payload["choices"][0]["message"]["content"]
            result = json.loads(content)
        except Exception as exc:
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} returned a malformed response: "
                f"{sanitize(exc)}"
            ) from exc
        if not isinstance(result, dict):
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} returned a malformed response: "
                f"expected an object, got {type(result).__name__}"
            )
        tracks = result.get("tracks")
        if not isinstance(tracks, list):
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} returned a malformed response: "
                f"'tracks' is {type(tracks).__name__}, expected a list"
            )
        # Keyed by id, because a guitar track may legitimately come back twice
        # (rhythm and lead) and the second entry must not replace the first.
        relabelled = {}
        for entry in tracks:
            track_id, instrument = self._entry_instrument(entry, instruments)
            relabelled.setdefault(track_id, []).append(instrument)
        missing = [i for i in range(len(instruments)) if i not in relabelled]
        if missing and not relabelled:
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} returned no usable track labels"
            )
        if missing:
            # Building the output only from the ids that came back would delete
            # the others: transcribed audio would vanish from the MIDI and the
            # tabs with nothing said about it.
            raise LabelingError(
                f"LLM endpoint {self.safe_endpoint} labelled {len(relabelled)} of "
                f"{len(instruments)} tracks; ids {missing} are missing"
            )
        # Every id is present -- incomplete coverage raised above -- so this
        # rebuilds the full track list in the caller's original order.
        out = []
        for i in range(len(instruments)):
            out.extend(relabelled[i])
        return out
