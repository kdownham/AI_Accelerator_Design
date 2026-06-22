#!/usr/bin/env python3
"""Translate the original FACET MAD-X line_4 lattice to Xsuite and plot Twiss."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xpart as xp
import xtrack as xt


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "MadX" / "FACET" / "flatGoldenLattice_line4_stub_10gev.madx"
OUT_PREFIX = ROOT / "facet_original_xsuite"


INITIAL_TWISS = {
    "betx": 29.515618,
    "bety": 45.174946,
    "alfx": 2.664132,
    "alfy": -3.582160,
    "dx": 0.0,
    "dpx": 0.0,
    "dy": 0.0,
    "dpy": 0.0,
}


def strip_comments(text: str) -> str:
    return re.sub(r"//.*", "", text)


def parse_value(body: str, key: str, default: float = 0.0) -> float:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*([^,;}}]+)", body, flags=re.IGNORECASE)
    if not match:
        return default
    return float(match.group(1).replace("D", "E"))


def parse_definitions(text: str) -> dict[str, tuple[str, str]]:
    definitions: dict[str, tuple[str, str]] = {}
    for match in re.finditer(
        r"(?ms)^([A-Za-z_][A-Za-z0-9_]*):\s*([A-Za-z0-9_]+)\b\s*(?:,(.*?))?;",
        text,
    ):
        name = match.group(1)
        kind = match.group(2).lower()
        body = match.group(3) or ""
        definitions[name.upper()] = (kind, body)
    return definitions


def parse_line(text: str, line_name: str = "line_4") -> list[str]:
    match = re.search(rf"(?ms)^{line_name}:\s*line\s*=\s*\((.*?)\);", text)
    if not match:
        raise ValueError(f"Could not find line {line_name!r}")
    body = strip_comments(match.group(1)).replace("\n", " ")
    return [name.strip() for name in body.split(",") if name.strip()]


def element_from_madx(name: str, kind: str, body: str) -> xt.BeamElement:
    length = parse_value(body, "l", 0.0)
    tilt = parse_value(body, "tilt", 0.0)

    if kind == "marker":
        return xt.Marker()
    if kind in {"drift", "collimator"}:
        return xt.Drift(length=max(length, 0.0))
    if kind == "quadrupole":
        kwargs = {"length": max(length, 0.0), "k1": parse_value(body, "k1", 0.0)}
        if tilt:
            kwargs["rot_s_rad"] = tilt
        return xt.Quadrupole(**kwargs)
    if kind == "sextupole":
        kwargs = {"length": max(length, 0.0), "k2": parse_value(body, "k2", 0.0)}
        if tilt:
            kwargs["rot_s_rad"] = tilt
        return xt.Sextupole(**kwargs)
    if kind == "sbend":
        kwargs = {
            "length": max(length, 0.0),
            "angle": parse_value(body, "angle", 0.0),
            "edge_entry_angle": parse_value(body, "e1", 0.0),
            "edge_exit_angle": parse_value(body, "e2", 0.0),
        }
        if tilt:
            kwargs["rot_s_rad"] = tilt
        return xt.Bend(**kwargs)
    if kind in {"hkicker", "vkicker", "multipole", "solenoid"}:
        # In this FACET source these elements in line_4 are zero-strength or
        # zero-length instrumentation/corrector placeholders. Preserve length.
        return xt.Drift(length=max(length, 0.0)) if length else xt.Marker()

    raise NotImplementedError(f"Unsupported MAD-X element type {kind!r} for {name}")


def build_line() -> tuple[xt.Line, Counter[str]]:
    text = SOURCE.read_text(encoding="utf-8")
    definitions = parse_definitions(text)
    names = parse_line(text, "line_4")

    elements = []
    element_names = []
    kinds: Counter[str] = Counter()

    for name in names:
        key = name.upper()
        if key not in definitions:
            raise KeyError(f"Missing element definition for {name}")
        kind, body = definitions[key]
        elements.append(element_from_madx(name, kind, body))
        element_names.append(name)
        kinds[kind] += 1

    line = xt.Line(elements=elements, element_names=element_names)
    line.particle_ref = xp.Particles(
        p0c=10e9,
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
    )
    return line, kinds


def twiss_to_frame(twiss) -> pd.DataFrame:
    columns = ["name", "s", "betx", "bety", "alfx", "alfy", "dx", "dy", "mux", "muy"]
    data = {}
    for col in columns:
        if hasattr(twiss, col):
            data[col] = list(getattr(twiss, col))
    return pd.DataFrame(data)


def plot_twiss(df: pd.DataFrame, line: xt.Line, kinds: Counter[str], out_png: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 3.0, 2.2]},
        constrained_layout=True,
    )
    survey_ax, beta_ax, disp_ax = axes

    s0 = 0.0
    for name, elem in zip(line.element_names, line.elements):
        length = float(getattr(elem, "length", 0.0) or 0.0)
        s1 = s0 + length
        cls = elem.__class__.__name__.lower()
        if "quadrupole" in cls:
            color = "#2563eb"
            height = 0.7
        elif "sextupole" in cls:
            color = "#9333ea"
            height = 0.5
        elif "bend" in cls:
            color = "#dc2626"
            height = 0.6
        else:
            color = "#737373"
            height = 0.2
        if length > 0:
            survey_ax.fill_between([s0, s1], [0, 0], [height, height], color=color, alpha=0.8, linewidth=0)
        s0 = s1

    survey_ax.set_ylim(0, 1)
    survey_ax.set_yticks([])
    survey_ax.set_ylabel("Lattice")
    survey_ax.set_title(
        "FACET original MAD-X line_4 translated to Xsuite "
        f"({sum(kinds.values())} elements)"
    )

    beta_ax.plot(df["s"], df["betx"], color="#0f766e", linewidth=1.8, label="BETX")
    beta_ax.plot(df["s"], df["bety"], color="#c2410c", linewidth=1.8, label="BETY")
    beta_ax.set_ylabel("Beta [m]")
    beta_ax.grid(True, alpha=0.3)
    beta_ax.legend(loc="upper left", ncol=2)

    disp_ax.plot(df["s"], df["dx"], color="#15803d", linewidth=1.5, label="DX")
    disp_ax.plot(df["s"], df["dy"], color="#be123c", linewidth=1.5, label="DY")
    disp_ax.axhline(0.0, color="#404040", linewidth=0.8, alpha=0.6)
    disp_ax.set_ylabel("Dispersion [m]")
    disp_ax.set_xlabel("S [m]")
    disp_ax.grid(True, alpha=0.3)
    disp_ax.legend(loc="upper left", ncol=2)

    end = df.iloc[-1]
    metrics = (
        f"End: BETX={end.betx:.6g} m, BETY={end.bety:.6g} m, "
        f"ALFX={end.alfx:.6g}, ALFY={end.alfy:.6g}\n"
        f"Max: BETX={df.betx.max():.6g} m, BETY={df.bety.max():.6g} m, "
        f"DX range=[{df.dx.min():.4g}, {df.dx.max():.4g}] m"
    )
    beta_ax.text(
        0.99,
        0.97,
        metrics,
        transform=beta_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d4d4d4", "alpha": 0.9},
    )

    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    line, kinds = build_line()
    line.build_tracker()
    twiss = line.twiss(method="4d", **INITIAL_TWISS)

    df = twiss_to_frame(twiss)
    csv_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_twiss.csv")
    json_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_line.json")
    png_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_twiss.png")
    summary_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.txt")

    df.to_csv(csv_path, index=False)
    line.to_json(json_path)
    plot_twiss(df, line, kinds, png_path)

    summary = [
        f"source={SOURCE}",
        f"elements={len(line.elements)}",
        "element_types=" + ", ".join(f"{k}:{v}" for k, v in sorted(kinds.items())),
        f"length_m={df['s'].iloc[-1]:.12g}",
        f"end_betx_m={df['betx'].iloc[-1]:.12g}",
        f"end_bety_m={df['bety'].iloc[-1]:.12g}",
        f"end_alfx={df['alfx'].iloc[-1]:.12g}",
        f"end_alfy={df['alfy'].iloc[-1]:.12g}",
        f"max_betx_m={df['betx'].max():.12g}",
        f"max_bety_m={df['bety'].max():.12g}",
        f"min_dx_m={df['dx'].min():.12g}",
        f"max_dx_m={df['dx'].max():.12g}",
        f"twiss_csv={csv_path}",
        f"line_json={json_path}",
        f"plot_png={png_path}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
