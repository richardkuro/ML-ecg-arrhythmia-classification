"""Scan all .py files for non-ASCII chars outside pure comment/docstring lines."""
import glob

files = sorted(glob.glob("*.py"))
found = []

for fn in files:
    with open(fn, encoding="utf-8") as f:
        lines = f.readlines()

    in_docstring = False
    dq = '"""'
    sq = "'''"

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track multiline docstrings
        if dq in line:
            count = line.count(dq)
            if count % 2 == 1:          # odd count -> toggle
                in_docstring = not in_docstring
        if sq in line:
            count = line.count(sq)
            if count % 2 == 1:
                in_docstring = not in_docstring

        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue

        for pos, ch in enumerate(line):
            if ord(ch) > 127:
                found.append((fn, i, pos, ch, line.rstrip()))
                break

if found:
    print(f"Found {len(found)} non-ASCII chars in executable code:")
    for fn, lno, pos, ch, line in found:
        print(f"  {fn}:{lno}  col {pos}  U+{ord(ch):04X} {repr(ch)}")
        print(f"    {line[:70]}")
else:
    print("No non-ASCII chars found in executable code lines.")
