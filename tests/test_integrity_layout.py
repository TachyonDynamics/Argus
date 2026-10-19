import hashlib

import pytest

from argus.integrity.layout import (list_blobs, list_manifests, manifest_blob_digests,
                                    models_dir)


def digest_name(data: bytes) -> str:
    return "sha256-" + hashlib.sha256(data).hexdigest()


def test_models_dir_prefers_an_explicit_override(tmp_path):
    (tmp_path / "override").mkdir()
    (tmp_path / "env").mkdir()
    assert models_dir(str(tmp_path / "override")) == tmp_path / "override"


def test_models_dir_falls_back_to_the_env_var(tmp_path, monkeypatch):
    (tmp_path / "env").mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "env"))
    assert models_dir() == tmp_path / "env"


def test_a_set_but_missing_path_does_not_fall_back_to_the_default_folder(tmp_path, monkeypatch):
    # Ollama would use (or create) the path OLLAMA_MODELS names, so quietly scanning
    # the default folder instead would report on the wrong models entirely.
    home = tmp_path / "home"
    (home / ".ollama" / "models").mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "nope"))
    assert models_dir() is None
    monkeypatch.delenv("OLLAMA_MODELS")
    assert models_dir() == home / ".ollama" / "models"     # default only when nothing is set
    assert models_dir(str(tmp_path / "also_nope")) is None  # an explicit override is exclusive too


def test_models_dir_returns_none_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no_home_ollama")
    assert models_dir() is None


def test_list_blobs_reads_digest_and_size_from_filename(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    data = b"hello world"
    name = digest_name(data)
    (blobs / name).write_bytes(data)
    found = list_blobs(tmp_path)
    assert len(found) == 1
    assert found[0].digest == name.removeprefix("sha256-")
    assert found[0].size == len(data)


def test_list_blobs_excludes_partial_downloads_and_unrelated_files(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / (digest_name(b"x") + "-partial")).write_bytes(b"x")
    (blobs / "readme.txt").write_text("not a blob")
    (blobs / "sha256-nothex").write_bytes(b"y")
    assert list_blobs(tmp_path) == []


def test_list_blobs_on_a_missing_blobs_dir_is_empty_not_an_error(tmp_path):
    assert list_blobs(tmp_path) == []


def test_list_blobs_records_mtime_for_the_verification_cache(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    f = blobs / digest_name(b"data")
    f.write_bytes(b"data")
    assert list_blobs(tmp_path)[0].mtime_ns == f.stat().st_mtime_ns


def test_list_manifests_uses_library_namespace_convention(tmp_path):
    p = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "8b"
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    found = list_manifests(tmp_path)
    assert [m.name for m in found] == ["llama3:8b"]
    assert found[0].path == p


def test_list_manifests_keeps_non_library_namespace_visible(tmp_path):
    p = tmp_path / "manifests" / "registry.ollama.ai" / "someuser" / "customtune" / "latest"
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    found = list_manifests(tmp_path)
    assert found[0].name == "someuser/customtune:latest"


def test_list_manifests_ignores_shallow_stray_files(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "stray.txt").write_text("x")
    assert list_manifests(tmp_path) == []


def test_list_manifests_on_a_missing_manifests_dir_is_empty_not_an_error(tmp_path):
    assert list_manifests(tmp_path) == []


def _manifest(tmp_path, body: bytes):
    p = tmp_path / "m"
    p.write_bytes(body)
    return p


H1, H2, H3 = "a" * 64, "b" * 64, "c" * 64


def test_manifest_blob_digests_reads_config_and_layers(tmp_path):
    body = (b'{"config":{"digest":"sha256:' + H1.encode() + b'"},'
            b'"layers":[{"digest":"sha256:' + H2.encode() + b'"},{"digest":"sha256:' + H3.encode() + b'"}]}')
    assert manifest_blob_digests(_manifest(tmp_path, body)) == [H1, H2, H3]


def test_manifest_blob_digests_dedupes_and_lowercases(tmp_path):
    body = (b'{"layers":[{"digest":"sha256:' + H1.upper().encode() + b'"},'
            b'{"digest":"sha256:' + H1.encode() + b'"}]}')
    assert manifest_blob_digests(_manifest(tmp_path, body)) == [H1]


def test_manifest_blob_digests_ignores_junk_entries(tmp_path):
    body = (b'{"config":"not a dict","layers":[{"digest":"sha256:short"},'
            b'{"digest":123},"nope",{"digest":"sha256:' + H2.encode() + b'"}]}')
    assert manifest_blob_digests(_manifest(tmp_path, body)) == [H2]


def test_manifest_blob_digests_survives_bad_json_or_wrong_shape(tmp_path):
    assert manifest_blob_digests(_manifest(tmp_path, b"{not json")) == []
    assert manifest_blob_digests(_manifest(tmp_path, b"[1,2,3]")) == []
    assert manifest_blob_digests(tmp_path / "missing") == []


def test_manifest_blob_digests_refuses_an_absurdly_large_manifest(tmp_path):
    p = tmp_path / "huge"
    p.write_bytes(b" " * (1024 * 1024 + 1))
    assert manifest_blob_digests(p) == []
