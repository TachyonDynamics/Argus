import asyncio

import httpx
import pytest

from argus.core.models import Outcome
from conftest import free_port
from fake_ollama import (ABORT, HANG, Script, chat_token, error_line, final, raw, sleep,
                         token)


def ms(trace, stamp):
    v = getattr(trace.stamps, stamp)
    return None if v is None else (v - trace.stamps.recv) / 1e6


async def test_streaming_is_byte_identical_and_timed(fake, argus):
    fake.script = Script([sleep(0.30), token("Hel"), sleep(0.10), token("lo"),
                          final(load_ns=111, prompt_ns=222, eval_ns=333, eval_count=2)])
    async with argus.client.stream("POST", "/api/generate",
                                   json={"model": "m", "prompt": "hi"}) as r:
        body = await r.aread()
    assert r.status_code == 200
    assert body == fake.sent                       # transparent: not a byte changed

    tr = await argus.trace()
    assert tr.outcome is Outcome.OK and tr.provider == "ollama"
    assert tr.request.model == "m" and tr.request.stream is True
    assert 300 <= ms(tr, "first_byte") < 450
    assert ms(tr, "first_out") == ms(tr, "first_byte")
    assert ms(tr, "first_answer") == ms(tr, "first_out")
    assert 400 <= ms(tr, "end") < 600
    assert tr.final.load_ns == 111 and tr.final.eval_count == 2
    assert tr.stamps.conn0 <= tr.stamps.conn1 <= tr.stamps.sent <= tr.stamps.hdr
    assert tr.stamps.hdr < tr.stamps.first_byte   # headers precede the first token
    assert tr.argus_ns < 20_000_000               # Argus's own per-request cost: under 20 ms


async def test_mid_stream_stall_shows_up_as_gap_not_client_wait(fake, argus):
    fake.script = Script([token("a"), sleep(0.5), token("b"), final()])
    async with argus.client.stream("POST", "/api/generate", json={"model": "m", "prompt": "x"}) as r:
        await r.aread()
    tr = await argus.trace()
    assert 500_000_000 <= tr.max_gap_ns < 800_000_000
    assert tr.max_gap_at_ns < 100_000_000         # the silence began right after chunk one
    assert tr.client_wait_ns < 100_000_000


async def test_non_streamed_generate(fake, argus):
    body = b'{"response":"hi","done":true,"done_reason":"stop","total_duration":9,"eval_count":2}'
    fake.script = Script([sleep(0.2), raw(body)], media_type="application/json")
    r = await argus.client.post("/api/generate", json={"model": "m", "prompt": "x", "stream": False})
    assert r.content == body
    tr = await argus.trace()
    assert tr.outcome is Outcome.OK and tr.request.stream is False
    assert 200 <= ms(tr, "first_out") < 350
    assert tr.final.total_ns == 9


async def test_chat_thinking_then_answer(fake, argus):
    fake.script = Script([
        ("line", {"message": {"role": "assistant", "content": "", "thinking": "hm"}, "done": False}),
        sleep(0.2), chat_token("Hi"), final()])
    async with argus.client.stream("POST", "/api/chat", json={"model": "m", "messages": []}) as r:
        await r.aread()
    tr = await argus.trace()
    assert ms(tr, "first_out") < 100 and ms(tr, "first_answer") >= 200


async def test_in_band_error_passes_through_and_is_classified(fake, argus):
    fake.script = Script([error_line("CUDA error: out of memory")])
    r = await argus.client.post("/api/generate", json={"model": "m", "prompt": "x"})
    assert r.status_code == 200 and b"out of memory" in r.content
    tr = await argus.trace()
    assert tr.outcome is Outcome.STREAM_ERROR and tr.error_code == "out_of_memory"


async def test_http_error_body_and_status_are_untouched(fake, argus):
    err = b'{"error":"model \'x\' not found"}'
    fake.script = Script([raw(err)], status=404, media_type="application/json")
    r = await argus.client.post("/api/chat", json={"model": "x", "messages": []})
    assert r.status_code == 404 and r.content == err
    tr = await argus.trace()
    assert tr.outcome is Outcome.HTTP_ERROR and tr.error_code == "model_not_found"
    assert tr.status_code == 404


async def test_upstream_dies_mid_stream_client_sees_truncation(fake, argus):
    fake.script = Script([token("a"), sleep(0.05), ABORT])
    with pytest.raises((httpx.RemoteProtocolError, httpx.ReadError)):
        async with argus.client.stream("POST", "/api/generate",
                                       json={"model": "m", "prompt": "x"}) as r:
            await r.aread()
    tr = await argus.trace()
    assert tr.outcome is Outcome.STREAM_TRUNCATED and tr.error_code == "stream_cut"


async def test_stream_that_ends_without_final_chunk_is_truncated(fake, argus):
    fake.script = Script([token("a"), token("b")])
    async with argus.client.stream("POST", "/api/generate", json={"model": "m", "prompt": "x"}) as r:
        await r.aread()
    tr = await argus.trace()
    assert tr.outcome is Outcome.STREAM_TRUNCATED and tr.error_code == "no_final_chunk"


async def test_upstream_unreachable_returns_502(make_argus):
    dead = free_port()                              # bound then released: nothing listens
    argus = await make_argus(f"http://127.0.0.1:{dead}")
    r = await argus.client.post("/api/generate", json={"model": "m", "prompt": "x"})
    assert r.status_code == 502 and "could not reach" in r.json()["error"]
    tr = await argus.trace()
    assert tr.outcome is Outcome.UPSTREAM_UNREACHABLE and tr.error_code == "connection_refused"


async def test_client_disconnect_mid_stream_cancels_upstream(fake, argus):
    fake.script = Script([token("a"), HANG])
    async with argus.client.stream("POST", "/api/generate", json={"model": "m", "prompt": "x"}) as r:
        async for _ in r.aiter_raw():
            break                                   # read one chunk, then walk away
    await fake.wait_cancelled()                     # Ollama's work was cancelled too
    tr = await argus.trace()
    assert tr.outcome is Outcome.CLIENT_CANCELLED


async def test_client_disconnect_while_waiting_for_first_token(fake, argus):
    fake.script = Script([sleep(30), token("late")])
    task = asyncio.create_task(
        argus.client.post("/api/generate", json={"model": "m", "prompt": "x"}, timeout=0.3))
    with pytest.raises(httpx.ReadTimeout):
        await task
    tr = await argus.trace()
    assert tr.outcome is Outcome.CLIENT_CANCELLED
    await fake.wait_cancelled()


async def test_concurrent_requests_record_queue_context(fake, argus):
    fake.script = Script([sleep(0.4), token("a"), final()])
    body = {"model": "m", "prompt": "x"}
    a = asyncio.create_task(argus.client.post("/api/generate", json=body))
    await asyncio.sleep(0.1)
    b = asyncio.create_task(argus.client.post("/api/generate", json=body))
    await asyncio.gather(a, b)
    first, second = await argus.trace(1), await argus.trace(2)
    assert (first.inflight_at_start, first.inflight_same_model) == (0, 0)
    assert (second.inflight_at_start, second.inflight_same_model) == (1, 1)


async def test_unbuffered_passthrough_for_large_chunked_upload(fake, argus):
    async def body():
        for _ in range(8):
            yield b"x" * 65536

    r = await argus.client.post("/api/blobs/sha256:abc", content=body())
    assert r.json() == {"received": 8 * 65536}
    tr = await argus.trace()
    assert tr.provider is None and tr.request is None and tr.outcome is Outcome.OK


async def test_headers_are_forwarded_faithfully(fake, argus):
    r = await argus.client.get("/echo", headers={"Origin": "http://example.test", "X-Test": "1",
                                                  "Connection": "keep-alive, X-Hop", "X-Hop": "drop"})
    assert r.headers.get_list("set-cookie") == ["a=1; Path=/", "b=2; Path=/"]   # duplicates kept
    assert r.headers["access-control-allow-origin"] == "http://example.test"
    assert r.headers["x-fake"] == "yes"
    seen = fake.received[-1]["headers"]
    assert seen["host"] == f"127.0.0.1:{fake.port}"       # Host rewritten to the upstream
    assert seen["origin"] == "http://example.test" and seen["x-test"] == "1"
    assert "x-hop" not in seen                             # named in Connection: hop-by-hop
    assert "gzip" in seen["accept-encoding"]               # non-inference: client's value kept


async def test_inference_requests_ask_upstream_for_identity_encoding(fake, argus):
    fake.script = Script([token("a"), final()])
    await argus.client.post("/api/generate", json={"model": "m", "prompt": "x"})
    assert fake.received[-1]["headers"]["accept-encoding"] == "identity"


async def test_get_traffic_is_traced_but_has_no_inference_fields(fake, argus):
    r = await argus.client.get("/api/tags")
    assert r.json() == {"models": []}
    tr = await argus.trace()
    assert tr.provider is None and tr.outcome is Outcome.OK and tr.method == "GET"