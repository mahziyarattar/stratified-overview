"""Patch benchmark-qed 0.4.0 so AutoQ can save assertion stats on Windows.

AutoQ writes stats to ``tempfile.NamedTemporaryFile(delete=True)`` and then
reopens the file by name, which Windows refuses while the file is open
(PermissionError). ``delete_on_close=False`` (Python 3.12+) keeps the file
deletable at context exit but closable for reopening. Idempotent.

    .venv-bench/Scripts/python benchmarks/patch_benchmark_qed_windows.py
"""

from __future__ import annotations

from pathlib import Path

import benchmark_qed

BROKEN = 'tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:'
FIXED = 'tempfile.NamedTemporaryFile(suffix=".json", delete=True, delete_on_close=False) as tmp:'


def main() -> None:
    target = Path(benchmark_qed.__file__).parent / "autoq" / "io" / "question.py"
    source = target.read_text(encoding="utf-8")
    count = source.count(BROKEN)
    if count:
        target.write_text(source.replace(BROKEN, FIXED), encoding="utf-8")
    print(f"patched {count} call(s) in {target}" if count else f"already patched: {target}")


if __name__ == "__main__":
    main()
