"""Adapter registry. Requests no adapter matches are passed through unparsed
(they still get a Trace with basic timing stamps)."""
from __future__ import annotations

from .base import ProviderAdapter
from .ollama import OllamaAdapter

_ADAPTERS: tuple[ProviderAdapter, ...] = (OllamaAdapter(),)


def match(method: str, path: str) -> ProviderAdapter | None:
    for adapter in _ADAPTERS:
        if adapter.matches(method, path):
            return adapter
    return None