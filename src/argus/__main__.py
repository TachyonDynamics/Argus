from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys
from pathlib import Path

from argus import __version__
from argus.app import run
from argus.config import Config


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,  # pytest and other harnesses may already hold the root logger
    )
    if not verbose:
        # httpx logs one INFO line per request (method, URL, status only - no
        # content), which just duplicates what the Requests page already shows.
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="argus", description="Local AI monitor for Ollama.")
    parser.add_argument("--upstream", help="where Ollama listens (default http://127.0.0.1:11435)")
    parser.add_argument("--proxy-port", type=int, help="port apps connect to (default 11434)")
    parser.add_argument("--admin-port", type=int, help="port of the web UI (default 47800)")
    parser.add_argument("--data-dir", type=Path, help="where Argus keeps its data")
    parser.add_argument("--verbose", action="store_true", help="log debugging detail")
    parser.add_argument("--version", action="version", version=f"argus {__version__}")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    try:
        cfg = Config.from_env()
        changes = {
            "upstream_url": args.upstream.rstrip("/") if args.upstream else None,
            "proxy_port": args.proxy_port,
            "admin_port": args.admin_port,
            "data_dir": args.data_dir,
        }
        cfg = dataclasses.replace(cfg, **{k: v for k, v in changes.items() if v is not None})
        cfg.validate()
    except ValueError as exc:
        print(f"Configuration problem: {exc}")
        return 2
    try:
        return asyncio.run(run(cfg))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())