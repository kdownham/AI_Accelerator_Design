#!/usr/bin/env python3
"""Translate the static 10 TeV FFS MAD-X line to Xsuite and plot Twiss."""

from __future__ import annotations

import ast
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
SOURCE_DIR = ROOT.parent.parent / "MadX" / "10TeV"
JOB = SOURCE_DIR / "job_10TeV_updated.madx"
FFS = SOURCE_DIR / "ffs_10TeV_updated.madx"
STRENGTHS = SOURCE_DIR / "optimized_strengths_updated.madx"
REFERENCE_TWISS = SOURCE_DIR / "bds_10TeV.twiss"
OUT_PREFIX = ROOT / "ffs_10tev_xsuite"


def strip_comments(text: str) -> str:
    return "\n".join(line.split("!")[0] for line in text.splitlines())


def statements(text: str) -> list[str]:
    return [part.strip() for part in strip_comments(text).split(";") if part.strip()]


def split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for ii, char in enumerate(text):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:ii].strip())
            start = ii + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


class ExpressionResolver:
    def __init__(self) -> None:
        self.expressions: dict[str, str] = {}
        self.cache: dict[str, float] = {}
        self.stack: set[str] = set()

    def add_assignments(self, text: str) -> None:
        for statement in statements(text):
            match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=)\s*(.+)", statement, flags=re.S)
            if not match:
                continue
            name, expression = match.groups()
            self.expressions[name.upper()] = expression.strip()
            self.cache.clear()

    def value(self, name: str) -> float:
        key = name.upper()
        if key in self.cache:
            return self.cache[key]
        if key in self.stack:
            raise ValueError(f"Circular MAD-X expression involving {name}")
        if key not in self.expressions:
            raise KeyError(f"Unknown MAD-X symbol {name}")
        self.stack.add(key)
        try:
            value = self.eval(self.expressions[key])
        finally:
            self.stack.remove(key)
        self.cache[key] = value
        return value

    def eval(self, expression: str) -> float:
        expression = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "E", expression)
        expression = expression.replace("^", "**")
        tree = ast.parse(expression, mode="eval")
        return float(self._eval_node(tree.body))

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return self.value(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            operations = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.Pow: lambda a, b: a**b,
            }
            for operation, fn in operations.items():
                if isinstance(node.op, operation):
                    return fn(left, right)
        raise ValueError(f"Unsupported MAD-X expression node: {ast.dump(node)}")


def parse_definitions(text: str) -> tuple[dict[str, tuple[str, dict[str, str]]], dict[str, str]]:
    elements: dict[str, tuple[str, dict[str, str]]] = {}
    lines: dict[str, str] = {}
    for statement in statements(text):
        match = re.match(
            r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\b(?:\s*:?=\s*|\s*,\s*)?(.*)\s*$",
            statement,
            flags=re.S,
        )
        if not match:
            continue
        name, kind, body = match.groups()
        body = body or ""
        kind = kind.upper()
        key = name.upper()
        if kind == "LINE":
            lines[key] = body.strip()
            continue
        if kind not in {"DRIFT", "SBEND", "QUADRUPOLE", "SEXTUPOLE", "MULTIPOLE", "MARKER"}:
            continue
        properties: dict[str, str] = {}
        for part in split_top_level(body):
            if not part:
                continue
            prop_match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:?=\s*(.+)", part, flags=re.S)
            if prop_match:
                prop_name, expression = prop_match.groups()
                properties[prop_name.upper()] = expression.strip()
        elements[key] = (kind, properties)
    return elements, lines


def line_tokens(body: str) -> list[tuple[int, str]]:
    body = body.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1].strip()
    tokens: list[tuple[int, str]] = []
    for part in split_top_level(body):
        match = re.fullmatch(r"\s*(\d+)\s*\*\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*", part)
        if match:
            tokens.append((int(match.group(1)), match.group(2).upper()))
        else:
            tokens.append((1, part.strip().upper()))
    return tokens


def expand_line(name: str, lines: dict[str, str], elements: dict[str, tuple[str, dict[str, str]]]) -> list[str]:
    key = name.upper()
    if key in elements:
        return [key]
    if key not in lines:
        raise KeyError(f"Unknown MAD-X line or element {name}")
    expanded: list[str] = []
    for count, token in line_tokens(lines[key]):
        item = expand_line(token, lines, elements)
        for _ in range(count):
            expanded.extend(item)
    return expanded


def property_value(properties: dict[str, str], name: str, resolver: ExpressionResolver, default: float = 0.0) -> float:
    expression = properties.get(name)
    return resolver.eval(expression) if expression is not None else default


def property_array(properties: dict[str, str], name: str, resolver: ExpressionResolver) -> list[float]:
    expression = properties.get(name)
    if expression is None:
        return []
    body = expression.strip().strip("{}")
    return [resolver.eval(part) for part in split_top_level(body)]


def make_element(kind: str, properties: dict[str, str], resolver: ExpressionResolver) -> xt.BeamElement:
    length = property_value(properties, "L", resolver)
    if kind == "MARKER":
        return xt.Marker()
    if kind == "DRIFT":
        return xt.Drift(length=length)
    if kind == "QUADRUPOLE":
        return xt.Quadrupole(length=length, k1=property_value(properties, "K1", resolver))
    if kind == "SEXTUPOLE":
        return xt.Sextupole(length=length, k2=property_value(properties, "K2", resolver))
    if kind == "SBEND":
        return xt.Bend(length=length, angle=property_value(properties, "ANGLE", resolver))
    if kind == "MULTIPOLE":
        return xt.Multipole(knl=property_array(properties, "KNL", resolver))
    raise ValueError(f"Unsupported element type {kind}")


def build_line() -> tuple[xt.Line, list[str], ExpressionResolver]:
    resolver = ExpressionResolver()
    ffs_text = FFS.read_text(encoding="utf-8")
    strength_text = STRENGTHS.read_text(encoding="utf-8")
    job_text = JOB.read_text(encoding="utf-8")

    # The call order in the MAD-X job is FFS definitions, optimized strengths,
    # then job-level scaling and Twiss assignments.
    resolver.add_assignments(ffs_text)
    resolver.add_assignments(strength_text)
    resolver.add_assignments(job_text)
    elements, lines = parse_definitions(ffs_text)
    active = expand_line("FFS", lines, elements)

    instances: list[xt.BeamElement] = []
    instance_names: list[str] = []
    seen: Counter[str] = Counter()
    for source_name in active:
        kind, properties = elements[source_name]
        seen[source_name] += 1
        instances.append(make_element(kind, properties, resolver))
        instance_names.append(f"{source_name.lower()}__{seen[source_name]:03d}")

    line = xt.Line(elements=instances, element_names=instance_names)
    line.particle_ref = xp.Particles(p0c=5_000e9, mass0=xp.ELECTRON_MASS_EV, q0=-1)
    return line, active, resolver


def read_tfs(path: Path) -> pd.DataFrame:
    columns: list[str] | None = None
    rows: list[list[str]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw_line.startswith("*"):
            columns = raw_line.split()[1:]
        elif columns and raw_line and not raw_line.startswith(("@", "$", "#")):
            parts = raw_line.split()
            if len(parts) >= len(columns):
                rows.append(parts[: len(columns)])
    if columns is None:
        raise ValueError(f"No TFS columns found in {path}")
    table = pd.DataFrame(rows, columns=columns)
    for column in table.columns:
        if column not in {"NAME", "KEYWORD", "PARENT", "COMMENTS"}:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def twiss_frame(twiss) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": list(twiss.name),
            "s": np.asarray(twiss.s),
            "betx": np.asarray(twiss.betx),
            "bety": np.asarray(twiss.bety),
            "alfx": np.asarray(twiss.alfx),
            "alfy": np.asarray(twiss.alfy),
            "dx": np.asarray(twiss.dx),
            "dy": np.asarray(twiss.dy),
        }
    )


def plot_twiss(table: pd.DataFrame, line: xt.Line, out_png: Path) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 3.0, 2.3]},
        constrained_layout=True,
    )
    survey_ax, beta_ax, disp_ax = axes

    s0 = 0.0
    for element in line.elements:
        length = float(getattr(element, "length", 0.0) or 0.0)
        s1 = s0 + length
        cls = element.__class__.__name__.lower()
        if "quadrupole" in cls:
            color, height = "#2563eb", 0.78
        elif "sextupole" in cls:
            color, height = "#9333ea", 0.56
        elif "bend" in cls:
            color, height = "#dc2626", 0.65
        elif "multipole" in cls:
            color, height = "#d97706", 0.46
        else:
            color, height = "#737373", 0.18
        if length > 0:
            survey_ax.fill_between([s0, s1], [0, 0], [height, height], color=color, linewidth=0)
        s0 = s1

    survey_ax.set_ylim(0, 1)
    survey_ax.set_yticks([])
    survey_ax.set_ylabel("Survey")
    survey_ax.set_title("10 TeV Flat-Beam FFS: Xsuite Twiss")

    beta_ax.plot(table["s"], table["betx"], color="#0f766e", linewidth=1.8, label="BETX")
    beta_ax.plot(table["s"], table["bety"], color="#c2410c", linewidth=1.8, label="BETY")
    beta_ax.set_ylabel("Beta [m]")
    beta_ax.grid(True, alpha=0.3)
    beta_ax.legend(loc="upper left", ncol=2)

    disp_ax.plot(table["s"], table["dx"], color="#15803d", linewidth=1.7, label="DX")
    disp_ax.plot(table["s"], table["dy"], color="#be123c", linewidth=1.7, label="DY")
    disp_ax.axhline(0, color="#404040", linewidth=0.8, alpha=0.6)
    disp_ax.set_xlabel("S [m]")
    disp_ax.set_ylabel("Dispersion [m]")
    disp_ax.grid(True, alpha=0.3)
    disp_ax.legend(loc="upper left", ncol=2)

    end = table.iloc[-1]
    beta_ax.text(
        0.99,
        0.97,
        f"End: BETX={end.betx:.6g} m, BETY={end.bety:.6g} m\\n"
        f"Max: BETX={table.betx.max():.6g} m, BETY={table.bety.max():.6g} m",
        transform=beta_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d4d4d4", "alpha": 0.9},
    )
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def compare_reference(xsuite: pd.DataFrame) -> dict[str, float]:
    if not REFERENCE_TWISS.exists():
        return {}
    madx = read_tfs(REFERENCE_TWISS)
    result = {"rows_xsuite": float(len(xsuite)), "rows_madx": float(len(madx))}
    madx_s = madx["S"].to_numpy()
    for xsuite_col, madx_col in [("s", "S"), ("betx", "BETX"), ("bety", "BETY"), ("alfx", "ALFX"), ("alfy", "ALFY"), ("dx", "DX"), ("dy", "DY")]:
        indexes = np.array([np.abs(madx_s - s).argmin() for s in xsuite["s"].to_numpy()])
        difference = xsuite[xsuite_col].to_numpy() - madx[madx_col].to_numpy()[indexes]
        result[f"max_abs_diff_{xsuite_col}"] = float(np.nanmax(np.abs(difference)))
        reference = madx[madx_col].to_numpy()[indexes]
        scale = max(float(np.nanmax(np.abs(reference))), 1e-30)
        result[f"max_rel_diff_{xsuite_col}"] = float(np.nanmax(np.abs(difference)) / scale)
    return result


def main() -> None:
    line, active, resolver = build_line()
    line.build_tracker()

    initial = {
        "betx": resolver.value("BETX"),
        "bety": resolver.value("BETY"),
        "alfx": resolver.value("ALFX"),
        "alfy": resolver.value("ALFY"),
        "dx": resolver.value("DX"),
        "dpx": 0.0,
        "dy": 0.0,
        "dpy": 0.0,
    }
    twiss = line.twiss(method="4d", **initial)
    table = twiss_frame(twiss)

    twiss_csv = OUT_PREFIX.with_name(OUT_PREFIX.name + "_twiss.csv")
    line_json = OUT_PREFIX.with_name(OUT_PREFIX.name + "_line.json")
    plot_png = OUT_PREFIX.with_name(OUT_PREFIX.name + "_twiss.png")
    summary_txt = OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.txt")
    table.to_csv(twiss_csv, index=False)
    line.to_json(line_json)
    plot_twiss(table, line, plot_png)

    comparison = compare_reference(table)
    end = table.iloc[-1]
    summary = [
        f"source_job={JOB}",
        f"source_ffs={FFS}",
        f"source_strengths={STRENGTHS}",
        f"active_line=FFS",
        f"energy_eV={5_000e9:.12g}",
        f"elements={len(line.elements)}",
        f"length_m={table.s.iloc[-1]:.12g}",
        f"initial_betx={initial['betx']:.12g}",
        f"initial_bety={initial['bety']:.12g}",
        f"end_betx={end.betx:.12g}",
        f"end_bety={end.bety:.12g}",
        f"end_alfx={end.alfx:.12g}",
        f"end_alfy={end.alfy:.12g}",
        f"end_dx={end.dx:.12g}",
        f"end_dy={end.dy:.12g}",
        f"max_betx={table.betx.max():.12g}",
        f"max_bety={table.bety.max():.12g}",
        f"twiss_csv={twiss_csv}",
        f"line_json={line_json}",
        f"plot_png={plot_png}",
    ]
    summary.extend(f"{key}={value:.12g}" for key, value in comparison.items())
    summary_txt.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
