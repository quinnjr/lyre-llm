"""Track labeling: the rule-based path, and the LLM that may refine it.

Two things are being pinned here. The first is that ``--llm`` never silently
downgrades: a user asks for LLM labeling precisely because rule-based labels are
not good enough, so a missing endpoint or a failed request is an error, not a
quiet fall back. The second is that nothing the remote endpoint sends can be
trusted -- not its response shape, not its redirects, and not the bytes in its
error bodies, which end up in a terminal and in a CI log.
"""

import json

import pytest

from lyre.errors import LabelingError
from lyre.instruments import Instrument, program_for
from lyre.labeling.llm import ALLOW_INSECURE_ENV, LLMLabeler, llm_from_env
from lyre.labeling.rules import label_rules, label_tracks
from lyre.tracking.hmm import Note

ENDPOINT = "https://api.example.com/v1"

SECRET = "sk-live-abcdef0123456789"
USERINFO_ENDPOINT = f"https://user:{SECRET}@api.example.com/v1/chat?token={SECRET}"

LLM_ENV = ("LYRE_LLM_ENDPOINT", "LYRE_LLM_KEY", "LYRE_LLM_MODEL")


def _notes(pitches, step=0.25):
    return [
        Note(pitch=p, start=i * step, end=(i + 1) * step, velocity=100.0)
        for i, p in enumerate(pitches)
    ]


def _one_instrument():
    return [Instrument(name="other", notes=_notes([60, 62]), source="guitar")]


def _two_instruments():
    return [
        Instrument(name="other", notes=_notes([60, 62]), source="guitar"),
        Instrument(name="other", notes=_notes([28, 33]), source="bass"),
    ]


@pytest.fixture
def no_llm_env(monkeypatch):
    """No LYRE_LLM_* in the environment.

    Explicit rather than assumed: a developer with LYRE_LLM_ENDPOINT exported in
    their shell would otherwise exercise a different code path than CI does, and
    the tests that matter most here are the ones about a *missing* endpoint.
    """
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)


# =====================================================================
# label_tracks: --llm must not silently downgrade
# =====================================================================


class _RaisingLLM:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def rename(self, instruments):
        self.calls += 1
        raise self.exc


@pytest.mark.parametrize(
    "exc",
    [
        LabelingError("LLM request to https://api.example/v1 failed: Connection refused"),
        LabelingError("LLM endpoint https://api.example/v1 returned HTTP 500 Server Error"),
        LabelingError("LLM endpoint https://api.example/v1 returned a malformed response"),
    ],
)
def test_strict_labeling_reraises_instead_of_downgrading(exc):
    llm = _RaisingLLM(exc)
    with pytest.raises(LabelingError) as caught:
        label_tracks(_one_instrument(), llm=llm, sink=[], strict=True)
    assert caught.value is exc
    assert llm.calls == 1


def test_non_strict_labeling_falls_back_but_records_the_reason():
    sink = []
    llm = _RaisingLLM(LabelingError("Connection refused"))
    out = label_tracks(_one_instrument(), llm=llm, sink=sink, strict=False)
    assert out and all(isinstance(i, Instrument) for i in out)
    assert len(sink) == 1
    assert "Connection refused" in sink[0]
    assert "rule-based" in sink[0]


def test_a_bug_inside_rename_is_not_masked_as_an_llm_failure():
    # Only LabelingError may be caught. A KeyError from a bug in rename() must
    # surface as itself, not as "LLM labeling failed".
    class Buggy:
        def rename(self, instruments):
            raise KeyError("tracks")

    sink = []
    with pytest.raises(KeyError):
        label_tracks(_one_instrument(), llm=Buggy(), sink=sink, strict=False)
    assert sink == []


def test_label_tracks_without_an_llm_uses_rules():
    sink = []
    out = label_tracks(
        [Instrument(name="other", notes=_notes([28, 33, 38]), source="bass")],
        llm=None,
        sink=sink,
    )
    assert sink == []
    assert [i.name for i in out] == ["bass"]


def test_an_unsplittable_guitar_stem_is_still_labelled_guitar():
    # Regression: _split_guitar used to return the ORIGINAL instrument when the
    # rhythm/lead split did not apply, so the guitar stem -- alone among the six
    # -- skipped the Instrument(...) rebuild and kept its incoming name on
    # program 0 (Acoustic Grand Piano).
    single_line = [
        Note(pitch=60 + i, start=i * 0.25, end=(i + 1) * 0.25, velocity=100.0)
        for i in range(4)
    ]
    out = label_rules([Instrument(name="vocals", notes=single_line, source="guitar")])
    assert [i.name for i in out] == ["guitar"]
    assert out[0].program == program_for("guitar")


# =====================================================================
# llm_from_env: --llm must not silently downgrade either
# =====================================================================


def test_llm_from_env_required_without_an_endpoint_is_an_error(no_llm_env):
    with pytest.raises(LabelingError) as exc:
        llm_from_env(required=True)
    message = str(exc.value)
    assert "LYRE_LLM_ENDPOINT" in message
    # The message has to say how to get out of the situation, in both directions.
    assert "--llm" in message


def test_llm_from_env_not_required_without_an_endpoint_returns_none(no_llm_env):
    # Nobody asked for it, so there is nothing to fail: label_tracks(llm=None)
    # is the rule-based path and is a perfectly good answer.
    assert llm_from_env(required=False) is None
    assert llm_from_env() is None


def test_llm_from_env_reads_the_endpoint_key_and_model(no_llm_env, monkeypatch):
    monkeypatch.setenv("LYRE_LLM_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("LYRE_LLM_KEY", SECRET)
    monkeypatch.setenv("LYRE_LLM_MODEL", "some-model")

    labeler = llm_from_env(required=True)

    assert labeler.endpoint == ENDPOINT
    assert labeler.api_key == SECRET
    assert labeler.model == "some-model"


def test_llm_from_env_defaults_the_model_and_tolerates_no_key(no_llm_env, monkeypatch):
    monkeypatch.setenv("LYRE_LLM_ENDPOINT", ENDPOINT)
    labeler = llm_from_env(required=True)
    assert labeler.api_key == ""
    assert labeler.model == "gpt-4o-mini"


def test_llm_from_env_still_validates_the_scheme(no_llm_env, monkeypatch):
    monkeypatch.setenv("LYRE_LLM_ENDPOINT", "http://api.example.com/v1")
    monkeypatch.delenv(ALLOW_INSECURE_ENV, raising=False)
    with pytest.raises(LabelingError):
        llm_from_env(required=True)


def test_asking_for_the_llm_without_an_endpoint_fails_the_run(no_llm_env):
    # The whole contract in one line: --llm with nothing configured must not
    # produce rule-based labels and a zero exit.
    from lyre.pipeline import Converter

    config = {
        "audio": {"sample_rate": 8000, "decode_channels": 2},
        "features": {
            "n_mels": 32, "n_fft": 512, "f_min": 30, "f_max": 8000,
            "window_frames": 16, "hop_ms": 10,
        },
        "model": {"channels": [4, 8], "n_notes": 128},
        "tracking": {},
        "labeling": {"use_llm": False},
        "inference": {"window_overlap": 0.5},
    }
    with pytest.raises(LabelingError):
        Converter(config, device="cpu", use_llm=True)

    config["labeling"]["use_llm"] = True
    with pytest.raises(LabelingError):
        Converter(config, device="cpu")


# =====================================================================
# Transport: scheme, credentials, redirects
# =====================================================================


def test_an_http_endpoint_is_rejected(monkeypatch):
    monkeypatch.delenv(ALLOW_INSECURE_ENV, raising=False)
    with pytest.raises(LabelingError) as exc:
        LLMLabeler(endpoint="http://api.example.com/v1", api_key="k")
    assert "https" in str(exc.value)
    assert ALLOW_INSECURE_ENV in str(exc.value)


def test_the_insecure_opt_out_permits_http(monkeypatch):
    monkeypatch.setenv(ALLOW_INSECURE_ENV, "1")
    labeler = LLMLabeler(endpoint="http://127.0.0.1:8080/v1", api_key="k")
    assert labeler.endpoint == "http://127.0.0.1:8080/v1"


def test_the_insecure_opt_out_does_not_cover_a_remote_host(monkeypatch):
    # "I am testing locally" never means "send my key across the network in
    # cleartext".
    monkeypatch.setenv(ALLOW_INSECURE_ENV, "1")
    with pytest.raises(LabelingError) as exc:
        LLMLabeler(endpoint="http://api.example.com/v1", api_key="k")
    assert "loopback" in str(exc.value)


@pytest.mark.parametrize(
    "endpoint", ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com"]
)
def test_non_http_schemes_are_rejected(endpoint, monkeypatch):
    # Even with the opt-out set: urllib would happily open file://, which turns
    # "call the LLM" into "read a local file".
    monkeypatch.setenv(ALLOW_INSECURE_ENV, "1")
    with pytest.raises(LabelingError):
        LLMLabeler(endpoint=endpoint, api_key="k")


def _fake_urlopen(monkeypatch, handler):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", handler)


def _http_error(body, code=500):
    import email.message
    import io
    import urllib.error

    return urllib.error.HTTPError(
        ENDPOINT, code, "Server Error", email.message.Message(), io.BytesIO(body)
    )


def test_the_endpoint_credential_never_reaches_an_error_message(monkeypatch):
    import urllib.error

    labeler = LLMLabeler(endpoint=USERINFO_ENDPOINT, api_key=SECRET)
    assert SECRET not in labeler.safe_endpoint

    _fake_urlopen(
        monkeypatch,
        lambda req, timeout=None: (_ for _ in ()).throw(
            urllib.error.URLError("Connection refused")
        ),
    )
    sink = []
    with pytest.raises(LabelingError) as exc:
        label_tracks(_one_instrument(), llm=labeler, sink=sink, strict=True)

    assert SECRET not in str(exc.value)
    assert "api.example.com" in str(exc.value)

    sink = []
    label_tracks(_one_instrument(), llm=labeler, sink=sink, strict=False)
    assert sink
    assert all(SECRET not in note for note in sink)


def test_the_endpoint_credential_never_reaches_the_scheme_error():
    with pytest.raises(LabelingError) as exc:
        LLMLabeler(endpoint=f"http://user:{SECRET}@api.example.com/v1?k={SECRET}")
    assert SECRET not in str(exc.value)
    assert "api.example.com" in str(exc.value)


def test_a_hostile_response_body_is_sanitised(monkeypatch):
    body = b"\x1b[2J\x1b[1;31mFATAL\r\nrm -rf /\x07 upstream said no"
    _fake_urlopen(
        monkeypatch, lambda req, timeout=None: (_ for _ in ()).throw(_http_error(body))
    )
    labeler = LLMLabeler(endpoint=ENDPOINT)
    with pytest.raises(LabelingError) as exc:
        labeler.rename(_one_instrument())

    message = str(exc.value)
    assert "HTTP 500" in message
    # The remote server controls this text; escapes in it would rewrite the
    # user's terminal or a CI log.
    for control in ("\x1b", "\r", "\n", "\x07"):
        assert control not in message
    assert "upstream said no" in message


def _capture_request(monkeypatch):
    seen = []

    def capture(req, timeout=None):
        seen.append(req)
        raise _http_error(b"stop here")

    _fake_urlopen(monkeypatch, capture)
    return seen


def test_no_authorization_header_when_the_key_is_empty(monkeypatch):
    seen = _capture_request(monkeypatch)
    with pytest.raises(LabelingError):
        LLMLabeler(endpoint=ENDPOINT, api_key="").rename(_one_instrument())
    headers = {k.lower(): v for k, v in seen[0].header_items()}
    assert "authorization" not in headers

    seen.clear()
    with pytest.raises(LabelingError):
        LLMLabeler(endpoint=ENDPOINT, api_key="k").rename(_one_instrument())
    headers = {k.lower(): v for k, v in seen[0].header_items()}
    assert headers["authorization"] == "Bearer k"


def test_the_key_is_an_unredirected_header(monkeypatch):
    # urllib's redirect handler copies req.headers onto the request it sends to
    # the redirect target, with no scheme and no host check: a 302 from a hostile
    # endpoint would otherwise walk the key out over cleartext to anywhere.
    # Unredirected headers are never copied.
    seen = _capture_request(monkeypatch)
    with pytest.raises(LabelingError):
        LLMLabeler(endpoint=ENDPOINT, api_key=SECRET).rename(_one_instrument())

    req = seen[0]
    assert req.unredirected_hdrs["Authorization"] == f"Bearer {SECRET}"
    assert "Authorization" not in req.headers
    assert SECRET not in repr(req.headers)


# =====================================================================
# rename: the response shape is untrusted input
# =====================================================================


class _FakeResponse:
    def __init__(self, body, url):
        self._body = body
        self.url = url

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _respond(monkeypatch, content, url=ENDPOINT):
    """Reply with ``content`` as the assistant message of a chat completion."""
    if not isinstance(content, str):
        content = json.dumps(content)
    envelope = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    _fake_urlopen(monkeypatch, lambda req, timeout=None: _FakeResponse(envelope, url))


def test_a_well_formed_response_relabels_every_track(monkeypatch):
    _respond(monkeypatch, {"tracks": [{"id": 0, "label": "piano"}, {"id": 1, "label": "bass"}]})
    out = LLMLabeler(endpoint=ENDPOINT).rename(_two_instruments())
    assert [i.name for i in out] == ["piano", "bass"]
    assert out[0].program == program_for("piano")
    # The notes are the transcription's, not the endpoint's.
    assert [n.pitch for n in out[0].notes] == [60, 62]


def test_two_entries_with_the_same_id_are_both_kept(monkeypatch):
    # A guitar track legitimately comes back twice, split into rhythm and lead;
    # keying by id and overwriting would silently drop half the performance.
    _respond(
        monkeypatch,
        {"tracks": [{"id": 0, "label": "guitar rhythm"}, {"id": 0, "label": "guitar lead"}]},
    )
    out = LLMLabeler(endpoint=ENDPOINT).rename(_one_instrument())
    assert [i.name for i in out] == ["guitar rhythm", "guitar lead"]


def test_an_entry_without_a_label_keeps_the_rule_based_name(monkeypatch):
    _respond(monkeypatch, {"tracks": [{"id": 0}]})
    out = LLMLabeler(endpoint=ENDPOINT).rename(
        [Instrument(name="bass", notes=_notes([28]), source="bass")]
    )
    assert [i.name for i in out] == ["bass"]


@pytest.mark.parametrize(
    "content,expected",
    [
        # Not an object at all.
        ("[]", "expected an object"),
        ("\"tracks\"", "expected an object"),
        # 'tracks' is the one key that is read, so its type is checked.
        ({"tracks": {}}, "'tracks' is dict"),
        ({"tracks": None}, "'tracks' is NoneType"),
        # An entry that is not an object has no id to read.
        ({"tracks": ["guitar"]}, "not an object"),
        # A missing id would default to track 0 and relabel the wrong track.
        ({"tracks": [{"label": "guitar"}]}, "'id' is not an integer"),
        ({"tracks": [{"id": "0", "label": "guitar"}]}, "'id' is not an integer"),
        # bool is an int in Python; "id": true must not select track 1.
        ({"tracks": [{"id": True, "label": "guitar"}]}, "'id' is not an integer"),
        # An id nobody sent cannot be indexed -- and a negative one would index
        # from the end and relabel a different track.
        ({"tracks": [{"id": 7, "label": "guitar"}]}, "only 1 tracks were sent"),
        ({"tracks": [{"id": -1, "label": "guitar"}]}, "only 1 tracks were sent"),
        # A non-string label would reach normalize_label and become "other".
        ({"tracks": [{"id": 0, "label": 42}]}, "non-string label"),
        ({"tracks": [{"id": 0, "label": ["guitar"]}]}, "non-string label"),
        # Nothing usable at all.
        ({"tracks": []}, "no usable track labels"),
        # Not JSON, and not the envelope shape.
        ("not json at all", "malformed response"),
    ],
)
def test_a_malformed_response_is_an_error(monkeypatch, content, expected):
    _respond(monkeypatch, content)
    with pytest.raises(LabelingError) as exc:
        LLMLabeler(endpoint=ENDPOINT).rename(_one_instrument())
    assert expected in str(exc.value)


def test_a_partially_labelled_response_is_an_error(monkeypatch):
    # Building the output only from the ids that came back would delete the
    # others: transcribed audio would vanish from the MIDI and the tabs with
    # nothing said about it.
    _respond(monkeypatch, {"tracks": [{"id": 0, "label": "guitar"}]})
    with pytest.raises(LabelingError) as exc:
        LLMLabeler(endpoint=ENDPOINT).rename(_two_instruments())
    message = str(exc.value)
    assert "1 of 2" in message
    assert "[1]" in message


def test_a_hostile_redirect_target_is_sanitised(monkeypatch):
    # The redirect target is quoted back to the user and the remote server chose
    # it, so it goes through the same sanitiser as everything else off the wire.
    _respond(
        monkeypatch,
        {"tracks": [{"id": 0, "label": "guitar"}]},
        url="https://elsewhere.example.net/v1\x1b[2Jrm -rf /\r\n",
    )
    with pytest.raises(LabelingError) as exc:
        LLMLabeler(endpoint=ENDPOINT).rename(_one_instrument())
    message = str(exc.value)
    for control in ("\x1b", "\r", "\n", "\x07"):
        assert control not in message
    assert "elsewhere.example.net" in message


def test_a_redirected_response_is_refused(monkeypatch):
    # The request body went to the redirect target, but the credential did not,
    # so whatever answered is not the endpoint the user configured.
    _respond(
        monkeypatch,
        {"tracks": [{"id": 0, "label": "guitar"}]},
        url="https://elsewhere.example.net/v1",
    )
    with pytest.raises(LabelingError) as exc:
        LLMLabeler(endpoint=ENDPOINT, api_key=SECRET).rename(_one_instrument())
    message = str(exc.value)
    assert "redirect" in message
    assert "elsewhere.example.net" in message
    assert SECRET not in message


def test_a_response_from_the_endpoint_itself_is_not_a_redirect(monkeypatch):
    _respond(monkeypatch, {"tracks": [{"id": 0, "label": "guitar"}]}, url=ENDPOINT)
    out = LLMLabeler(endpoint=ENDPOINT).rename(_one_instrument())
    assert [i.name for i in out] == ["guitar"]


def test_a_response_with_no_url_is_not_treated_as_a_redirect(monkeypatch):
    # Not every response object carries a url; absence must not be read as
    # "redirected somewhere else".
    envelope = json.dumps(
        {"choices": [{"message": {"content": json.dumps({"tracks": [{"id": 0, "label": "bass"}]})}}]}
    ).encode()

    class _NoUrl(_FakeResponse):
        def __init__(self, body):
            self._body = body

    _fake_urlopen(monkeypatch, lambda req, timeout=None: _NoUrl(envelope))
    out = LLMLabeler(endpoint=ENDPOINT).rename(_one_instrument())
    assert [i.name for i in out] == ["bass"]
