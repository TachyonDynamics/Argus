"""Unit tests for forwarder.py helpers that don't need a running server."""
import socket

import httpx
import pytest

from argus.core.models import Outcome
from argus.proxy.forwarder import classify_exception


def chain(inner: BaseException, outer: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException as e:
            raise outer from e
    except BaseException as caught:
        return caught


def test_connection_refused_at_the_top_level():
    assert classify_exception(ConnectionRefusedError(), streaming=False) == \
        (Outcome.UPSTREAM_UNREACHABLE, "connection_refused")


def test_connection_refused_wrapped_by_httpx_is_found_via_the_cause_chain():
    wrapped = chain(ConnectionRefusedError(), httpx.ConnectError("connect failed"))
    assert classify_exception(wrapped, streaming=False) == \
        (Outcome.UPSTREAM_UNREACHABLE, "connection_refused")


def test_dns_failure_wrapped_by_httpx():
    wrapped = chain(socket.gaierror("nope"), httpx.ConnectError("connect failed"))
    assert classify_exception(wrapped, streaming=False) == \
        (Outcome.UPSTREAM_UNREACHABLE, "dns_failure")


def test_bare_connect_error_without_a_recognised_cause_is_connect_failed():
    assert classify_exception(httpx.ConnectError("x"), streaming=False) == \
        (Outcome.UPSTREAM_UNREACHABLE, "connect_failed")


def test_connect_timeout_is_reported_as_a_timeout_not_a_bare_connect_error():
    # httpx.ConnectTimeout is a subclass of both ConnectError and TimeoutException;
    # the timeout classification must win.
    assert classify_exception(httpx.ConnectTimeout("x"), streaming=False) == \
        (Outcome.UPSTREAM_TIMEOUT, "timeout")


def test_read_error_maps_differently_for_streaming_vs_non_streaming():
    exc = httpx.ReadError("x")
    assert classify_exception(exc, streaming=True) == (Outcome.STREAM_TRUNCATED, "stream_cut")
    assert classify_exception(exc, streaming=False) == (Outcome.UPSTREAM_RESET, "connection_reset")


def test_remote_protocol_error_counts_as_a_reset_class():
    assert classify_exception(httpx.RemoteProtocolError("x"), streaming=True) == \
        (Outcome.STREAM_TRUNCATED, "stream_cut")


def test_unrecognised_exception_becomes_proxy_error_with_its_type_name():
    assert classify_exception(ValueError("boom"), streaming=False) == \
        (Outcome.PROXY_ERROR, "ValueError")


def test_long_cause_chain_still_finds_the_os_error_within_the_search_depth():
    exc = ConnectionRefusedError()
    for _ in range(5):
        exc = chain(exc, RuntimeError("wrapping"))
    assert classify_exception(exc, streaming=False) == \
        (Outcome.UPSTREAM_UNREACHABLE, "connection_refused")