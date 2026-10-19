import orjson

from argus.core.models import RequestInfo
from argus.proxy.adapters import match
from argus.proxy.adapters.ollama import OllamaAdapter, classify_error_text

A = OllamaAdapter()


def req(kind="generate", stream=True):
    return RequestInfo(kind=kind, stream=stream)


def lines(*objs):
    return b"".join(orjson.dumps(o) + b"\n" for o in objs)


def test_registry_matches_only_ollama_inference_posts():
    assert match("POST", "/api/generate")
    assert match("POST", "/api/chat/")
    assert match("POST", "/api/embed")
    assert match("GET", "/api/generate") is None
    assert match("POST", "/api/pull") is None
    assert match("GET", "/api/tags") is None


def test_parse_request_defaults_and_options():
    r = A.parse_request(orjson.dumps({"model": "llama3", "prompt": "x",
                                      "options": {"num_ctx": 8192}}), "/api/generate")
    assert (r.kind, r.model, r.stream, r.num_ctx, r.parse_ok) == ("generate", "llama3", True, 8192, True)
    r = A.parse_request(orjson.dumps({"model": "m", "messages": [], "stream": False}), "/api/chat")
    assert (r.kind, r.stream) == ("chat", False)
    r = A.parse_request(orjson.dumps({"model": "m", "input": "x"}), "/api/embed")
    assert (r.kind, r.stream) == ("embed", False)


def test_parse_request_survives_bad_json():
    r = A.parse_request(b"{not json", "/api/chat")
    assert r.parse_ok is False and r.kind == "chat" and r.prompt_bytes == 9


def test_stream_parser_handles_chunks_split_mid_line():
    p = A.new_stream_parser(req(), 200)
    data = lines({"response": "Hel", "done": False},
                 {"response": "", "done": True, "done_reason": "stop", "total_duration": 5,
                  "load_duration": 2, "prompt_eval_count": 7, "prompt_eval_duration": 1,
                  "eval_count": 3, "eval_duration": 2})
    cut = 20
    first = p.feed(data[:cut])
    assert not first.has_output           # no complete line yet
    second = p.feed(data[cut:])
    assert second.has_output and second.has_answer
    p.finish()
    assert p.final.load_ns == 2 and p.final.eval_count == 3 and p.final.done_reason == "stop"
    assert p.error_code is None


def test_stream_parser_thinking_is_output_but_not_answer():
    p = A.new_stream_parser(req("chat"), 200)
    info = p.feed(lines({"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False}))
    assert info.has_output and not info.has_answer
    info = p.feed(lines({"message": {"role": "assistant", "content": "Hi"}, "done": False}))
    assert info.has_output and info.has_answer


def test_stream_parser_tool_calls_count_as_output():
    p = A.new_stream_parser(req("chat"), 200)
    info = p.feed(lines({"message": {"role": "assistant", "content": "",
                                     "tool_calls": [{"function": {"name": "f"}}]}, "done": False}))
    assert info.has_output and not info.has_answer


def test_in_band_error_is_classified_without_keeping_text():
    p = A.new_stream_parser(req(), 200)
    p.feed(lines({"error": "CUDA error: out of memory"}))
    assert p.error_code == "out_of_memory" and p.final is None


def test_error_body_without_trailing_newline_is_read_on_finish():
    p = A.new_stream_parser(req(), 404)
    p.feed(b'{"error":"model \'x\' not found"}')
    assert p.error_code is None
    p.finish()
    assert p.error_code == "model_not_found"


def test_non_streamed_reply_parses_on_finish_and_marks_first_byte_as_output():
    p = A.new_stream_parser(req(stream=False), 200)
    body = orjson.dumps({"response": "hi", "done": True, "total_duration": 9, "eval_count": 2})
    info = p.feed(body[:10])
    assert info.has_output and info.has_answer
    p.feed(body[10:])
    assert p.final is None
    p.finish()
    assert p.final.total_ns == 9 and p.final.eval_count == 2


def test_embed_reply_has_final_stats_but_no_answer():
    p = A.new_stream_parser(req("embed", stream=False), 200)
    info = p.feed(orjson.dumps({"embeddings": [[0.1]], "total_duration": 4, "load_duration": 1}))
    assert info.has_output and not info.has_answer
    p.finish()
    assert p.final.total_ns == 4 and p.final.load_ns == 1


def test_garbage_lines_are_ignored():
    p = A.new_stream_parser(req(), 200)
    p.feed(b"not json\n[1,2]\n\n")
    p.finish()
    assert p.final is None and p.error_code is None


def test_error_classification_order():
    assert classify_error_text("llama runner process has terminated: CUDA error: out of memory") == "out_of_memory"
    assert classify_error_text("llama runner process has terminated: exit status 0xc0000409") == "runner_crashed"
    assert classify_error_text("model requires more system memory (12 GiB) than is available") == "out_of_memory"
    assert classify_error_text("model 'nope' not found, try pulling it first") == "model_not_found"
    assert classify_error_text("something odd") == "unclassified"