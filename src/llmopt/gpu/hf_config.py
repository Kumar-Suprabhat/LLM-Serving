"""Fetching a model's ``config.json`` with nothing but the standard library.

The VRAM budget is derived from model geometry, so the geometry has to be known *before* the
engine is launched -- and ideally before heavyweight dependencies are even installed. Using
``urllib`` rather than ``huggingface_hub`` means preflight works on a bare interpreter, which
keeps the dependency graph of the safety path as small as possible.

Only ``config.json`` is fetched: a few kilobytes, rather than the 16 GB of weights.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

from llmopt.errors import ConfigError, DatasetUnavailableError

__all__ = ["fetch_model_config", "load_model_config"]

_HF_RESOLVE_TEMPLATE: Final = "https://huggingface.co/{model}/resolve/{revision}/config.json"
_TIMEOUT_SECONDS: Final = 30
_MAX_CONFIG_BYTES: Final = 1 << 20
"""A config.json is a few KB; refuse anything suspiciously large rather than reading it all."""


def fetch_model_config(
    model_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Download a model's ``config.json`` from the Hugging Face Hub.

    Args:
        model_id: Repo id, e.g. ``"Qwen/Qwen3-8B"``.
        revision: Branch, tag or commit SHA. Defaults to ``main``; **pin a SHA before
            publishing numbers**, since geometry changes would silently invalidate a budget.
        token: Access token for gated repos.

    Returns:
        The parsed config.

        Raises:
        ConfigError: If ``model_id`` is empty or the response is not a JSON object.
        DatasetUnavailableError: If the repo or file cannot be fetched -- gated, renamed,
            misspelled, or the network is unavailable.
    """
    if not model_id.strip():
        raise ConfigError("model_id cannot be empty.")

    url = _HF_RESOLVE_TEMPLATE.format(model=model_id.strip("/"), revision=revision or "main")
    request = urllib.request.Request(url, headers={"User-Agent": "llmopt/0.1"})  # noqa: S310
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw = response.read(_MAX_CONFIG_BYTES + 1)
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code == 401:
            hint = " The repo appears gated; set HF_TOKEN and accept the licence on the Hub."
        elif exc.code == 404:
            hint = " Check the repo id and revision spelling."
        raise DatasetUnavailableError(
            f"Could not fetch config.json for {model_id!r} at revision "
            f"{revision or 'main'!r}: HTTP {exc.code}.{hint}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DatasetUnavailableError(
            f"Could not reach the Hugging Face Hub for {model_id!r}: {exc}"
        ) from exc

    if len(raw) > _MAX_CONFIG_BYTES:
        raise ConfigError(
            f"config.json for {model_id!r} exceeds {_MAX_CONFIG_BYTES} bytes, which is not a "
            f"plausible model config."
        )

    try:
        config = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"config.json for {model_id!r} is not valid JSON: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"config.json for {model_id!r} is not a JSON object.")
    return config


def load_model_config(
    model: str,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Load a model config from a local directory if present, else from the Hub.

    Preferring a local path keeps repeat sessions offline once weights are cached, and makes the
    budget reproducible from a checkout without network access.

    Args:
        model: Repo id or local directory containing ``config.json``.
        revision: Revision to fetch when going to the Hub.
        token: Access token for gated repos.

    Returns:
        The parsed config.

    Raises:
        ConfigError: If a local config exists but is malformed.
        DatasetUnavailableError: If neither a local nor a remote config can be read.
    """
    local = Path(model)
    candidate = local / "config.json" if local.is_dir() else local
    if candidate.is_file():
        try:
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ConfigError(f"Local config at {candidate} is unreadable: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ConfigError(f"Local config at {candidate} is not a JSON object.")
        return parsed

    return fetch_model_config(model, revision=revision, token=token)
