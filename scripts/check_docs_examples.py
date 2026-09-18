"""Run documentation Python blocks in a fresh namespace per page.

Uses a temporary working directory for example outputs. Optional titled blocks
are included with --include-optional; benchmark documents are never executed.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ.setdefault("MPLBACKEND", "Agg")
    pages = [root / "README.md"] + sorted(
        path for path in (root / "docs").rglob("*.md")
        if "benchmarks" not in path.relative_to(root / "docs").parts
    )
    pattern = re.compile(r"^```python([^\n]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
    count = 0
    checked = 0
    skipped = 0
    previous = Path.cwd()
    try:
        with TemporaryDirectory(prefix="clifra-doc-examples-") as temporary:
            os.chdir(temporary)
            for path in pages:
                source = path.read_text(encoding="utf-8")
                namespace = {"__name__": "__main__"}
                blocks = 0
                for match in pattern.finditer(source):
                    if "Optional" in match[1] and not args.include_optional:
                        skipped += 1
                        continue
                    # Preserve document line numbers in tracebacks.
                    code = "\n" * source.count("\n", 0, match.start(2)) + match[2]
                    exec(compile(code, str(path), "exec"), namespace)
                    blocks += 1
                if blocks:
                    checked += 1
                    count += blocks
                    print(f"PASS {path.relative_to(root)} ({blocks} blocks)", flush=True)
            if "matplotlib.pyplot" in sys.modules:
                pyplot = sys.modules["matplotlib.pyplot"]
                for number in pyplot.get_fignums():
                    pyplot.figure(number).canvas.draw()
                pyplot.close("all")
    finally:
        os.chdir(previous)
    print(f"Passed {count} blocks across {checked} pages; skipped {skipped} optional blocks.")


if __name__ == "__main__":
    main()
