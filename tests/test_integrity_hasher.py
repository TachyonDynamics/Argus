import hashlib

from argus.integrity.hasher import sha256_file


def test_matches_hashlib_for_a_small_file(tmp_path):
    data = b"the quick brown fox"
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()


def test_matches_hashlib_across_multiple_chunks(tmp_path):
    # A tiny chunk size forces several reads; the result must be identical to
    # hashing it all in one go.
    data = bytes(range(256)) * 50   # 12,800 bytes
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    assert sha256_file(p, chunk_size=7) == hashlib.sha256(data).hexdigest()


def test_empty_file(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()


def test_missing_file_returns_none_not_an_exception(tmp_path):
    assert sha256_file(tmp_path / "does_not_exist") is None


def test_a_directory_passed_by_mistake_returns_none_not_an_exception(tmp_path):
    d = tmp_path / "a_dir"
    d.mkdir()
    assert sha256_file(d) is None


def test_cancellation_stops_hashing_between_chunks(tmp_path):
    import threading

    import pytest

    from argus.integrity.hasher import ScanCancelled

    p = tmp_path / "big.bin"
    p.write_bytes(b"x" * 1000)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ScanCancelled):
        sha256_file(p, chunk_size=10, cancel=cancel)


def test_an_unset_cancel_event_changes_nothing(tmp_path):
    import threading
    data = b"abc" * 100
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    assert sha256_file(p, chunk_size=7, cancel=threading.Event()) == hashlib.sha256(data).hexdigest()
