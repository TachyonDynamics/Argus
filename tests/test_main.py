import logging

from argus.__main__ import configure_logging, main


def test_configure_logging_quiets_httpx_by_default():
    configure_logging(verbose=False)
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_verbose_leaves_httpx_alone():
    logging.getLogger("httpx").setLevel(logging.NOTSET)
    configure_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.NOTSET


def test_bad_config_from_cli_is_reported_not_raised(capsys):
    rc = main(["--proxy-port", "9999", "--admin-port", "9999",
              "--upstream", "http://127.0.0.1:9999"])
    assert rc == 2
    assert "not available" not in capsys.readouterr().out  # validate() catches this first