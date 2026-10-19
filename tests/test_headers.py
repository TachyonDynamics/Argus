from argus.proxy.headers import filter_headers, request_headers, response_headers


def test_filter_headers_strips_fixed_hop_by_hop_set():
    raw = [(b"Connection", b"keep-alive"), (b"Content-Type", b"application/json"),
          (b"Transfer-Encoding", b"chunked"), (b"Upgrade", b"websocket")]
    out = filter_headers(raw)
    assert out == [(b"Content-Type", b"application/json")]


def test_filter_headers_strips_headers_named_in_connection():
    raw = [(b"Connection", b"keep-alive, X-Custom"), (b"X-Custom", b"secret"),
          (b"X-Keep", b"yes")]
    out = filter_headers(raw)
    assert out == [(b"X-Keep", b"yes")]


def test_filter_headers_drop_is_case_insensitive():
    raw = [(b"HOST", b"example.com"), (b"X-Keep", b"1")]
    assert filter_headers(raw, drop=[b"host"]) == [(b"X-Keep", b"1")]


def test_filter_headers_preserves_duplicate_keys_and_original_casing():
    raw = [(b"Set-Cookie", b"a=1"), (b"Set-Cookie", b"b=2")]
    assert filter_headers(raw) == raw


def test_request_headers_drops_host_and_adds_identity_only_for_inference():
    raw = [(b"Host", b"old"), (b"Accept-Encoding", b"gzip"), (b"X-Test", b"1")]
    out = request_headers(raw, buffered=False, identity=True)
    assert (b"Host", b"old") not in out
    # HTTP header names are case-insensitive; the code always emits lowercase here.
    assert (b"accept-encoding", b"identity") in out
    assert sum(1 for k, _ in out if k.lower() == b"accept-encoding") == 1  # replaced, not duplicated
    assert (b"X-Test", b"1") in out

    passthrough = request_headers(raw, buffered=False, identity=False)
    assert (b"Accept-Encoding", b"gzip") in passthrough  # non-inference: client's own value kept


def test_request_headers_drops_content_length_only_when_buffered():
    raw = [(b"Content-Length", b"42")]
    assert request_headers(raw, buffered=True, identity=False) == []
    assert request_headers(raw, buffered=False, identity=False) == raw


def test_request_headers_still_applies_hop_by_hop_rules():
    raw = [(b"Host", b"x"), (b"Connection", b"close")]
    assert request_headers(raw, buffered=False, identity=False) == []


def test_response_headers_is_plain_passthrough_minus_hop_by_hop():
    raw = [(b"Content-Type", b"application/json"), (b"Connection", b"close"),
          (b"Access-Control-Allow-Origin", b"*")]
    assert response_headers(raw) == [(b"Content-Type", b"application/json"),
                                     (b"Access-Control-Allow-Origin", b"*")]