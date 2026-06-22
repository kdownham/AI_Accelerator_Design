#!/usr/bin/env python3
"""Build a FACET MAD-X deck that uses MATCH for the FF/D local-beta tune."""

from __future__ import annotations

import re
from pathlib import Path

from optimize_facet_less_focusing import BRHO_TM
from prepare_facet_madx_run import SOURCE, convert_lcavity


BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "facet_ffd_madx_match.madx"
TWISS_OUTPUT = BASE_DIR / "facet_ffd_madx_match_twiss.tfs"

FAMILIES: list[tuple[str, str, float]] = [
    ("Q5FF", "kq5ff", 55.5435),
    ("Q4FF", "kq4ff", 62.4475),
    ("Q3FF", "kq3ff", 63.9877),
    ("Q2FF", "kq2ff", 23.3828),
    ("Q1FF", "kq1ff", 35.9843),
    ("Q0FF", "kq0ff", 23.3828),
    ("Q0D", "kq0d", 23.9000),
    ("Q1D", "kq1d", 38.6000),
    ("Q2D", "kq2d", 22.3000),
]

# Targets from the previous FF/D solution, now imposed with MAD-X MATCH.
TARGETS = {
    "DEX20_10": {
        "betx": 1.792857701,
        "bety": 1.979471573,
        "alfx": 1.336908398,
        "alfy": 0.4387401769,
    },
    "DEX20_11": {
        "betx": 1.588902324,
        "bety": 1.913128696,
        "alfx": 1.212533817,
        "alfy": 0.3905457817,
    },
    "#e": {
        "betx": 39.83160324,
        "bety": 3.089824673,
        "alfx": -2.225679971,
        "alfy": -0.159813506,
    },
}


def current_k1(text: str, element_name: str) -> float:
    pattern = re.compile(
        rf"(?ms)^{re.escape(element_name)}:\s*quadrupole,\s*(?P<body>.*?);"
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"could not find quadrupole {element_name}")
    k1_match = re.search(r"\bk1\s*=\s*([^,;]+)", match.group("body"), flags=re.IGNORECASE)
    if not k1_match:
        raise ValueError(f"could not find k1 for quadrupole {element_name}")
    return float(k1_match.group(1))


def replace_k1_with_variable(text: str, element_name: str, variable_name: str) -> str:
    pattern = re.compile(
        rf"(?ms)^({re.escape(element_name)}:\s*quadrupole,\s*)(?P<body>.*?);"
    )

    def repl(match: re.Match[str]) -> str:
        body = re.sub(
            r"\bk1\s*=\s*[^,;]+",
            f"k1 := {variable_name}",
            match.group("body"),
            count=1,
            flags=re.IGNORECASE,
        )
        return f"{match.group(1)}{body};"

    new_text, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise ValueError(f"could not update quadrupole {element_name}")
    return new_text


def build_variable_block(source_text: str) -> str:
    lines = ["// FF/D match variables initialized from the original lattice;"]
    for element_name, variable_name, _ in FAMILIES:
        lines.append(f"{variable_name} = {current_k1(source_text, element_name):.12g};")
    return "\n".join(lines)


def insert_variable_block(text: str, variable_block: str) -> str:
    marker = "// Bmad lattice file: flatGoldenLattice.bmad;"
    if marker in text:
        return text.replace(marker, f"{marker}\n\n{variable_block}", 1)
    return f"{variable_block}\n\n{text}"


def build_match_block() -> str:
    lines = [
        "",
        "// Re-optimize the FF/D section using MAD-X MATCH.",
        "match, sequence=lat, beta0=initial;",
    ]
    for _, variable_name, cap_tm in FAMILIES:
        cap_k1 = cap_tm / BRHO_TM
        lines.append(
            f"  vary, name={variable_name}, step=1.0e-5, "
            f"lower={-cap_k1:.12g}, upper={cap_k1:.12g};"
        )
    for location, constraints in TARGETS.items():
        for column, value in constraints.items():
            lines.append(f"  constraint, range={location}, {column}={value:.12g};")
    lines.extend(
        [
            "  lmdif, calls=12000, tolerance=1.0e-12;",
            "endmatch;",
            "",
            "value, kq5ff, kq4ff, kq3ff, kq2ff, kq1ff, kq0ff, kq0d, kq1d, kq2d;",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    text = re.sub(
        r"(?ms)^(?P<name>[A-Za-z_][A-Za-z0-9_]*):\s*lcavity,\s*(?P<body>.*?);",
        convert_lcavity,
        source_text,
    )
    for element_name, variable_name, _ in FAMILIES:
        text = replace_k1_with_variable(text, element_name, variable_name)

    variable_block = build_variable_block(source_text)
    text = insert_variable_block(text, variable_block)
    text = text.replace(
        "select, flag=twiss, column=name,keyword,s,l,k1l,tilt,betx,bety,alfx,alfy,dx,dy,mux,muy;\n"
        'twiss, beta0 = initial, file="facet_twiss.tfs";',
        "",
    )
    text = text.replace(
        "twiss, beta0 = initial;",
        build_match_block()
        + 'select, flag=twiss, column=name,keyword,s,l,k1l,tilt,betx,bety,alfx,alfy,dx,dy,mux,muy;\n'
        + f'twiss, beta0 = initial, file="{TWISS_OUTPUT}";',
    )
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
