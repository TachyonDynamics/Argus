from pathlib import Path

from argus.integrity.models import (BLOB_CORRUPT, BLOB_OK, BLOB_UNREADABLE, MANIFEST_APPROVED,
                                    MANIFEST_CHANGED, MANIFEST_MISSING, MANIFEST_NEW,
                                    BlobFinding, ManifestFinding, ScanResult)
from argus.ui.present_integrity import clip, present_integrity, short

NS = 1_000_000_000
NOW = 100 * NS


def blob(digest="a" * 64, actual=None, status=BLOB_OK, size=1024, models=(), path="p",
        rehashed=True):
    return BlobFinding(digest=digest, path=Path(path), size=size, status=status,
                       actual_digest=actual if actual is not None else digest,
                       models=list(models), verified_wall_ns=NOW, rehashed=rehashed)


def manifest(name="m:1", status=MANIFEST_APPROVED, current="a" * 64, approved="a" * 64,
            approved_wall_ns=NOW - 60 * NS, blob_digests=()):
    return ManifestFinding(name=name, path=Path("p"), status=status, current_digest=current,
                           approved_digest=approved, approved_wall_ns=approved_wall_ns,
                           blob_digests=list(blob_digests))


def result(**kw):
    kw.setdefault("scanned_wall_ns", NOW)
    kw.setdefault("models_dir", Path("/models"))
    return ScanResult(**kw)


def test_no_scan_yet():
    v = present_integrity(None, scanning=False, now_wall_ns=NOW)
    assert v["state"] == "unscanned" and "haven't been checked" in v["headline"]
    assert v["last_checked"] == ""


def test_no_scan_yet_but_one_is_running():
    v = present_integrity(None, scanning=True, now_wall_ns=NOW)
    assert "Checking" in v["headline"] and v["subline"] == "" and v["can_edit"] is False


def test_no_models_directory():
    v = present_integrity(result(models_dir=None), scanning=False, now_wall_ns=NOW)
    assert v["state"] == "no_models" and "OLLAMA_MODELS" in v["subline"]


def test_models_directory_exists_but_is_empty():
    v = present_integrity(result(), scanning=False, now_wall_ns=NOW)
    assert v["state"] == "empty"


def test_all_clean():
    r = result(blobs=[blob()], manifests=[manifest(blob_digests=["a" * 64])])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    assert v["state"] == "clean" and v["warnings"] == []
    assert v["models"][0]["status_label"] == "Approved"
    assert "1 minute ago" in v["models"][0]["approved"]


def test_corrupt_blob_shows_metadata_only_never_the_bytes():
    r = result(blobs=[blob(status=BLOB_CORRUPT, actual="b" * 64, models=["llama3:8b"], size=5_368_709_120)])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    w = v["warnings"][0]
    assert w["expected"] == ("a" * 64)[:12] and w["found"] == ("b" * 64)[:12]
    assert w["size"] == "5.0 GB" and w["subject"] == "llama3:8b"
    assert "ollama rm llama3:8b" in w["fix"] and "ollama pull llama3:8b" in w["fix"]
    assert v["state"] == "warnings" and "1 model file problem" in v["headline"]


def test_corrupt_blob_used_by_several_models_lists_all_and_skips_the_named_command():
    r = result(blobs=[blob(status=BLOB_CORRUPT, actual="b" * 64, models=["a:1", "b:2"])])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    assert v["warnings"][0]["subject"] == "a:1, b:2"
    fix = v["warnings"][0]["fix"]
    assert "ollama rm a:1" not in fix and "ollama rm b:2" not in fix


def test_corrupt_blob_with_no_owning_model_gets_a_cautious_note_not_a_command():
    r = result(blobs=[blob(status=BLOB_CORRUPT, actual="b" * 64, models=[])])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    assert "ollama rm" not in v["warnings"][0]["fix"]
    assert "left over" in v["warnings"][0]["fix"]


def test_unreadable_blob_is_a_review_item_not_a_warning():
    r = result(blobs=[blob(status=BLOB_UNREADABLE, actual=None, models=["m:1"])])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    assert v["warnings"] == [] and v["state"] == "clean"
    assert v["review_unreadable"][0]["models"] == "m:1"


def test_new_model_is_review_only_not_a_warning():
    r = result(manifests=[manifest(status=MANIFEST_NEW, approved=None, approved_wall_ns=None)])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    assert v["warnings"] == [] and v["state"] == "clean"
    assert v["review_new"][0]["name"] == "m:1"


def test_changed_model_is_a_warning_with_both_digests_and_a_relative_time():
    r = result(manifests=[manifest(status=MANIFEST_CHANGED, current="c" * 64,
                                   approved="a" * 64, approved_wall_ns=NOW - 3 * 86400 * NS)])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    w = v["warnings"][0]
    assert w["approved"] == ("a" * 64)[:12] and w["found"] == ("c" * 64)[:12]
    assert "3 days ago" in w["approved_when"]
    assert v["state"] == "warnings"


def test_missing_model_is_review_only_and_can_be_forgotten_by_name():
    r = result(manifests=[manifest(status=MANIFEST_MISSING, current=None)])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    assert v["warnings"] == [] and v["review_missing"][0]["raw_name"] == "m:1"


def test_model_row_summarises_file_coverage():
    r = result(blobs=[blob(digest="a" * 64), blob(digest="b" * 64, status=BLOB_CORRUPT, actual="z" * 64)],
              manifests=[manifest(blob_digests=["a" * 64, "b" * 64, "c" * 64])])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    files = v["models"][0]["files"]
    assert "1 of 3 files failed" in files and "1 not on disk" in files


def test_coverage_line_distinguishes_full_scan_from_cached():
    r = result(blobs=[blob(rehashed=False)])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    assert "already verified" in v["coverage"]
    r2 = result(blobs=[blob(rehashed=True)], forced=True)
    assert "checked just now" in present_integrity(r2, scanning=False, now_wall_ns=NOW)["coverage"]


def test_untrusted_model_name_is_clipped_before_it_ever_reaches_a_template():
    long_name = "x" * 500 + ":latest"
    r = result(manifests=[manifest(name=long_name, status=MANIFEST_NEW, approved=None,
                                   approved_wall_ns=None)])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    assert len(v["review_new"][0]["name"]) <= 121
    assert v["review_new"][0]["raw_name"] == long_name    # the real name is kept for the form post


def test_a_name_that_looks_like_shell_injection_never_reaches_the_suggested_command():
    payload = "m:1; rm -rf ~"
    r = result(blobs=[blob(status=BLOB_CORRUPT, actual="b" * 64, models=[payload])])
    v = present_integrity(r, scanning=False, now_wall_ns=NOW)
    assert payload not in v["warnings"][0]["fix"]
    assert "rm -rf" not in v["warnings"][0]["fix"]


def test_notice_messages_render_from_known_keys():
    assert "already running" in present_integrity(result(), scanning=False, now_wall_ns=NOW,
                                                  msg="busy")["notice"]
    assert present_integrity(result(), scanning=False, now_wall_ns=NOW,
                             msg="approved_all", msg_count=3)["notice"] == "Approved 3 models."
    assert present_integrity(result(), scanning=False, now_wall_ns=NOW,
                             msg="approved_all", msg_count=1)["notice"] == "Approved 1 model."


def test_short_and_clip_helpers():
    assert short(None) == "–" and short("a" * 64) == "a" * 12
    assert clip("short") == "short" and clip("x" * 200).endswith("…") and len(clip("x" * 200)) == 120
