from argus.core.models import RequestInfo, Trace
from argus.ui.present import LIVE_STALL_NS, one_line_why, present_live

NS = 1_000_000_000


def trace_at(recv=0, sent=None, hdr=None, first_byte=None, last_chunk=None, model="m",
            kind="generate", client_addr="127.0.0.1:5000"):
    t = Trace("POST", "/api/generate", client_addr=client_addr)
    t.stamps.recv = recv
    t.stamps.sent = sent
    t.stamps.hdr = hdr
    t.stamps.first_byte = first_byte
    t.stamps.last_chunk = last_chunk
    t.provider = "ollama"
    t.request = RequestInfo(kind=kind, model=model, stream=True)
    return t


def test_phase_progression_from_connecting_to_generating():
    assert present_live(trace_at(), now_ns=100)["phase"] == "Connecting"
    assert present_live(trace_at(sent=10), now_ns=100)["phase"] == "Waiting for Ollama"
    assert present_live(trace_at(sent=10, hdr=20), now_ns=100)["phase"] == "Starting"
    t = trace_at(sent=10, hdr=20, first_byte=30, last_chunk=30)
    assert present_live(t, now_ns=100)["phase"] == "Generating"


def test_phase_becomes_paused_after_a_long_silence():
    t = trace_at(sent=0, hdr=1, first_byte=2, last_chunk=2)
    assert present_live(t, now_ns=2 + LIVE_STALL_NS - 1)["phase"] == "Generating"
    assert present_live(t, now_ns=2 + LIVE_STALL_NS + 1)["phase"] == "Paused"


def test_present_live_reports_model_kind_client_and_elapsed():
    t = trace_at(recv=1 * NS, sent=1 * NS, hdr=int(1.1 * NS), model="llama3", kind="chat",
                client_addr="127.0.0.1:9999")
    c = present_live(t, now_ns=int(3.5 * NS))
    assert c["model"] == "llama3" and c["kind"] == "chat"
    assert c["client"] == "from 127.0.0.1:9999"
    assert c["elapsed"] == "2.5 s" and c["elapsed_ns"] == int(2.5 * NS)


def test_present_live_handles_a_request_with_no_parsed_model_yet():
    t = Trace("POST", "/api/generate")
    t.provider = "ollama"  # request body not parsed (bad JSON): request stays None
    c = present_live(t, now_ns=1000)
    assert c["model"] == "–" and c["kind"] == "other"
    assert c["client"] == "from an unknown address"


class _Fake:
    def __init__(self, primary=None):
        self.primary = primary


class _Finding:
    def __init__(self, title):
        self.title = title


def test_one_line_why_prefers_the_diagnosis_when_present():
    row = {"outcome": "ok"}
    assert one_line_why(row, _Fake(_Finding("Slower than usual"))) == "Slower than usual"


def test_one_line_why_falls_back_to_a_plain_outcome_label():
    assert one_line_why({"outcome": "ok"}, _Fake(None)) == "Completed normally"
    assert one_line_why({"outcome": "upstream_unreachable"}, _Fake(None)) == "Couldn't reach Ollama"
