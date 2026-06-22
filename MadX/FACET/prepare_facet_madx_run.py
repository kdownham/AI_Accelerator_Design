#!/usr/bin/env python3
"""Prepare the FACET Bmad-exported lattice for a MAD-X TWISS run."""

from __future__ import annotations

import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "flatGoldenLattice_line4_stub_10gev.madx"
OUTPUT = BASE_DIR / "facet_twiss_run.madx"


def convert_lcavity(match: re.Match[str]) -> str:
    name = match.group("name")
    body = match.group("body")
    length_match = re.search(r"\bl\s*=\s*([^,;]+)", body, flags=re.IGNORECASE)
    length = length_match.group(1).strip() if length_match else "0"
    return f"{name}: drift, l = {length};"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    text = re.sub(
        r"(?ms)^(?P<name>[A-Za-z_][A-Za-z0-9_]*):\s*lcavity,\s*(?P<body>.*?);",
        convert_lcavity,
        text,
    )
    text = text.replace(
        "twiss, beta0 = initial;",
        'select, flag=twiss, column=name,keyword,s,l,k1l,tilt,betx,bety,alfx,alfy,dx,dy,mux,muy;\n'
        'twiss, beta0 = initial, file="facet_twiss.tfs";',
    )
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
