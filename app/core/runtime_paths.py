"""Resolve writable runtime paths for local and serverless deployments."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, os.PathLike[str]]
SERVERLESS_ENV_VARS = (
    "VERCEL",
    "AWS_LAMBDA_FUNCTION_NAME",
    "AWS_EXECUTION_ENV",
)


def _is_serverless() -> bool:
    return any(os.getenv(name) for name in SERVERLESS_ENV_VARS)


def _prepare_writable_directory(path: Path) -> Optional[Path]:
    """Create ``path`` and prove it is writable before returning it."""
    probe = path / f".nova-write-probe-{uuid.uuid4().hex}"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return path
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def get_runtime_directory(
    purpose: str,
    *,
    env_var: Optional[str] = None,
    local_directory: Optional[PathLike] = None,
) -> Path:
    """Return a writable directory without assuming the source tree is mutable.

    Explicit configuration wins. Serverless platforms use the operating
    system's temporary directory, while local development keeps the existing
    project-relative locations when they are writable.
    """
    candidates = []

    if env_var and os.getenv(env_var):
        candidates.append(Path(os.environ[env_var]).expanduser())

    temporary_directory = Path(tempfile.gettempdir()) / "nova_pulse" / purpose

    if _is_serverless():
        candidates.append(temporary_directory)
        if local_directory is not None:
            candidates.append(Path(local_directory))
    else:
        if local_directory is not None:
            candidates.append(Path(local_directory))
        candidates.append(temporary_directory)

    for candidate in candidates:
        prepared = _prepare_writable_directory(candidate.resolve())
        if prepared is not None:
            return prepared

    raise RuntimeError(f"No writable runtime directory is available for {purpose}")
