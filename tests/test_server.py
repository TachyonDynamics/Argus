import uvicorn

from argus.server import Server, make_server


async def _app(scope, receive, send):
    pass


def test_transparent_server_omits_its_own_headers_and_x_forwarded_handling():
    s = make_server(_app, host="127.0.0.1", port=0, transparent=True)
    assert isinstance(s, Server)
    assert s.config.server_header is False
    assert s.config.date_header is False
    assert s.config.proxy_headers is False


def test_admin_server_keeps_normal_header_behaviour():
    s = make_server(_app, host="127.0.0.1", port=0, transparent=False)
    assert s.config.server_header is True
    assert s.config.date_header is True
    assert s.config.proxy_headers is True


def test_common_settings_regardless_of_transparency():
    s = make_server(_app, host="127.0.0.1", port=0)
    assert s.config.access_log is False
    assert s.config.lifespan == "off"
    assert s.config.ws == "none"
    assert s.config.timeout_graceful_shutdown == 5


def test_server_does_not_install_its_own_signal_handlers():
    # Two servers share one process (see app._install_stop); if either installed
    # its own handler it would silently steal Ctrl+C from the other.
    s = make_server(_app, host="127.0.0.1", port=0)
    assert s.install_signal_handlers() is None      # uvicorn < 0.30 hook: a no-op
    with s.capture_signals():                        # uvicorn >= 0.30 hook: also a no-op
        pass


def test_make_server_returns_a_real_uvicorn_config():
    s = make_server(_app, host="127.0.0.1", port=12345)
    assert isinstance(s.config, uvicorn.Config)
    assert s.config.host == "127.0.0.1" and s.config.port == 12345