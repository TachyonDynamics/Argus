import pytest

from argus.config import Config, default_data_dir


def test_defaults_and_derived_paths(tmp_path):
    cfg = Config(data_dir=tmp_path)
    assert cfg.upstream_url == "http://127.0.0.1:11435"
    assert cfg.proxy_port == 11434 and cfg.admin_port == 47800
    assert cfg.db_path == tmp_path / "argus.db"
    assert not (tmp_path / "sub").exists()
    Config(data_dir=tmp_path / "sub").ensure_dirs()
    assert (tmp_path / "sub").is_dir()


@pytest.mark.parametrize("host,local", [
    ("127.0.0.1", True), ("localhost", True), ("[::1]", True),
    ("10.0.0.5", False), ("example.com", False), ("0.0.0.0", False),
])
def test_upstream_is_local(host, local):
    assert Config(upstream_url=f"http://{host}:11435").upstream_is_local is local


def test_validate_rejects_non_http_scheme_and_missing_host():
    with pytest.raises(ValueError, match="ARGUS_UPSTREAM"):
        Config(upstream_url="ftp://127.0.0.1:11435").validate()
    with pytest.raises(ValueError, match="ARGUS_UPSTREAM"):
        Config(upstream_url="not a url").validate()


def test_validate_rejects_port_collision_only_when_upstream_is_local():
    with pytest.raises(ValueError, match="share one port"):
        Config(upstream_url="http://127.0.0.1:11434", proxy_port=11434).validate()
    Config(upstream_url="http://example.com:11434", proxy_port=11434).validate()  # remote: fine


def test_validate_infers_default_port_from_scheme():
    with pytest.raises(ValueError, match="share one port"):
        Config(upstream_url="https://127.0.0.1", proxy_port=443).validate()


def test_from_env_uses_defaults_when_unset():
    cfg = Config.from_env({})
    assert cfg == Config(data_dir=cfg.data_dir)  # only data_dir varies (default_data_dir())


def test_from_env_applies_overrides_and_strips_trailing_slash(tmp_path):
    env = {
        "ARGUS_UPSTREAM": "http://127.0.0.1:9000/",
        "ARGUS_PROXY_HOST": "0.0.0.0",
        "ARGUS_PROXY_PORT": "8080",
        "ARGUS_ADMIN_PORT": "8081",
        "ARGUS_DATA_DIR": str(tmp_path),
        "ARGUS_MAX_BODY": "1024",
    }
    cfg = Config.from_env(env)
    assert cfg.upstream_url == "http://127.0.0.1:9000"   # trailing slash stripped
    assert cfg.proxy_host == "0.0.0.0" and cfg.proxy_port == 8080
    assert cfg.admin_port == 8081 and cfg.data_dir == tmp_path
    assert cfg.max_buffered_body == 1024


def test_from_env_still_validates(monkeypatch):
    with pytest.raises(ValueError, match="share one port"):
        Config.from_env({"ARGUS_UPSTREAM": "http://127.0.0.1:11434", "ARGUS_PROXY_PORT": "11434"})


def test_default_data_dir_prefers_localappdata(monkeypatch, tmp_path):
    # A plain (non-Windows-syntax) path so the check works under the Path flavour
    # of whatever OS runs the tests; the real separator handling is Windows's own.
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_data_dir() == tmp_path / "Argus"
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert default_data_dir().name == "Argus"


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(AttributeError):
        cfg.proxy_port = 1