from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


AUTO_INSTALL_ENV = "INVESTIGATION_SEARCH_AUTO_INSTALL"


@dataclass(frozen=True)
class InstallPlan:
    requirements_files: tuple[Path, ...] = ()
    packages: tuple[str, ...] = ()


def auto_install_enabled(flag: bool | None = None) -> bool:
    if flag is not None:
        return bool(flag)
    raw = os.environ.get(AUTO_INSTALL_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def ensure_installed(
    *,
    requirements_files: Sequence[str | Path] | None = None,
    packages: Sequence[str] | None = None,
    auto_install: bool | None = None,
    upgrade: bool = False,
    quiet: bool = False,
) -> InstallPlan:
    """Best-effort dependency installer.

    - If auto_install is not enabled, it is a no-op.
    - If requirements files exist, prefer installing them.
    - Otherwise fallback to package spec list.
    """
    if not auto_install_enabled(auto_install):
        return InstallPlan()

    req_files: list[Path] = []
    for rf in list(requirements_files or []):
        p = Path(rf)
        if p.exists():
            req_files.append(p)

    pkgs = [str(p).strip() for p in list(packages or []) if str(p).strip()]

    if req_files:
        _pip_install_requirements(req_files, upgrade=upgrade, quiet=quiet)
        return InstallPlan(requirements_files=tuple(req_files))

    if pkgs:
        _pip_install_packages(pkgs, upgrade=upgrade, quiet=quiet)
        return InstallPlan(packages=tuple(pkgs))

    return InstallPlan()


def repo_root() -> Path | None:
    """Best-effort discovery for a repo checkout (for requirements files)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "requirements.txt").exists():
            return parent
    return None


def requirements_path(name: str) -> Path | None:
    root = repo_root()
    if root is None:
        return None
    p = root / name
    return p if p.exists() else None


def _pip_install_requirements(files: Iterable[Path], *, upgrade: bool, quiet: bool) -> None:
    for path in files:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(path)]
        if upgrade:
            cmd.append("--upgrade")
        if quiet:
            cmd.append("--quiet")
        subprocess.run(cmd, check=True)


def _pip_install_packages(packages: Sequence[str], *, upgrade: bool, quiet: bool) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    if upgrade:
        cmd.append("--upgrade")
    if quiet:
        cmd.append("--quiet")
    subprocess.run(cmd, check=True)
