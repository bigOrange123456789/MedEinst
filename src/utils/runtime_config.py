"""Small path and endpoint helpers for the public launchers."""

import os
from pathlib import Path
from urllib.parse import urlparse
import ipaddress


def config_bool(value, default: bool = False) -> bool:
    """Parse YAML booleans and lab2-style string flags without bool('false')."""
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"Invalid boolean setting: {value!r}")
    return bool(value)


def is_loopback_url(url: str) -> bool:
    """Only allow automatic dummy API keys for local HTTP services."""
    host = urlparse(url).hostname
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def configured_base_url(section: dict):
    """An optional environment override takes precedence over the YAML URL."""
    variable = section.get("base_url_env")
    return (os.getenv(variable) if variable else None) or section.get("base_url") or None

def config_relative_path(value: str, config_path: Path) -> str:
    """Resolve a YAML path relative to its configuration file."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())
