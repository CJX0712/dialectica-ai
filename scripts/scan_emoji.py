"""P0 gate: no emoji in source, docs or UI.

Emoji are acceptable in prose; they are not acceptable as functional icons,
because they render inconsistently across terminals, editors and CI logs.

Detection uses `ord()` code-point ranges only -- the source file itself must
never contain an emoji literal, or it gets mangled by the console code page.

Author: 晨星
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".pytest_cache"}
EXTENSIONS = {".py", ".md", ".html", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".txt"}

RANGES = [
    (0x1F000, 0x1FAFF),  # emoji & symbols
    (0x2600, 0x27BF),    # misc symbols & dingbats
    (0x2190, 0x21FF),    # arrows
    (0x2B00, 0x2BFF),
    (0xFE00, 0xFE0F),    # variation selectors
    (0x1F1E6, 0x1F1FF),  # regional indicators
]

ALLOW = {(0x2190, 0x21FF)}  # arrows are used legitimately in docs


def is_emoji(code: int) -> bool:
    for lo, hi in RANGES:
        if (lo, hi) in ALLOW:
            continue
        if lo <= code <= hi:
            return True
    return False


def scan() -> list[str]:
    findings: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in EXTENSIONS:
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        for ch in line:
                            if is_emoji(ord(ch)):
                                findings.append(
                                    f"{os.path.relpath(path, ROOT)}:{lineno} U+{ord(ch):04X}"
                                )
                                break
            except (UnicodeDecodeError, OSError):
                continue
    return findings


def main() -> int:
    findings = scan()
    if findings:
        print(f"FAIL: 发现 {len(findings)} 处 emoji：")
        for f in findings[:50]:
            print("  " + f)
        return 1
    print("PASS: 全仓无 emoji（功能图标一律不使用 emoji）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
