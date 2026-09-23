"""Regenerate requirements.lock.txt from the *current* interpreter.

Usage:
    python -m venv .venv
    .venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt
    .venv/Scripts/python scripts/make_lock.py

The lock is the exact closure of what was installed, minus platform-specific
packages that would break installation on other operating systems
(`win32-setctime` / `pywin32` are Windows-only).

Author: 晨星
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "requirements.lock.txt")
# Normalised on the fly: pip freeze may print either `win32-setctime` or
# `win32_setctime` depending on version, and matching on the raw string
# silently lets the Windows-only package through.
DROP = {"win32-setctime", "pywin32", "pip", "setuptools", "wheel", "pkg-resources"}


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def main() -> int:
    raw = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==")[0].split("@")[0].strip()
        if _normalize(name) in DROP:
            continue
        lines.append(line)
    lines.sort(key=lambda s: s.lower())
    header = [
        "# Dialectica 完整依赖闭包（精确版本）。",
        "# 由 scripts/make_lock.py 从干净虚拟环境 `pip freeze` 生成 —— 天然「已验证可安装」。",
        "# 已剔除 Windows 专有包（win32-setctime / pywin32），保证 Linux / macOS 也可一键安装。",
        "#",
        "# 复现方式：",
        "#   python -m venv .venv",
        "#   .venv/Scripts/pip install -r requirements.lock.txt",
        "#",
        "# 作者: 晨星",
        "",
    ]
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(header + lines) + "\n")
    print(f"wrote {len(lines)} pinned packages to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
