#!/usr/bin/env python3
"""Create an HTML summary comparing FACET tracking with SR modes."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_HTML = ROOT / "facet_sr_tracking_comparison_summary.html"
PLOT = ROOT / "facet_sr_phase_space_rms_compare.png"

RUNS = {
    "No SR": ROOT / "facet_realistic_independent_snapshots.npz",
    "Mean SR": ROOT / "facet_realistic_radiation_mean_start_end.npz",
    "Quantum SR": ROOT / "facet_realistic_radiation_quantum_start_end.npz",
}

COORDS = [
    ("x", "m"),
    ("px", "rad"),
    ("y", "m"),
    ("py", "rad"),
    ("zeta", "m"),
    ("delta", ""),
]


def end_array(data, label: str, coord: str) -> np.ndarray:
    if label == "No SR":
        return data[coord][-1].astype(float)
    return data[f"end_{coord}"].astype(float)


def end_state(data, label: str) -> np.ndarray:
    if label == "No SR":
        return data["state"][-1] > 0
    return data["end_state"] > 0


def rms(values: np.ndarray, alive: np.ndarray) -> float:
    return float(np.std(values[alive]))


def mean(values: np.ndarray, alive: np.ndarray) -> float:
    return float(np.mean(values[alive]))


def fmt(value: float) -> str:
    return f"{value:.10e}"


def collect() -> list[dict[str, object]]:
    rows = []
    for label, path in RUNS.items():
        data = np.load(path, allow_pickle=True)
        alive = end_state(data, label)
        row: dict[str, object] = {
            "mode": label,
            "path": path,
            "alive": int(alive.sum()),
            "lost": int((~alive).sum()),
        }
        for coord, _unit in COORDS:
            values = end_array(data, label, coord)
            row[f"rms_{coord}"] = rms(values, alive)
            row[f"mean_{coord}"] = mean(values, alive)
        rows.append(row)
    return rows


def table(rows: list[dict[str, object]]) -> str:
    headers = ["Mode", "Alive", "Lost"] + [f"RMS {c} [{u}]" if u else f"RMS {c}" for c, u in COORDS]
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{html.escape(h)}</th>" for h in headers)
    out.append("</tr></thead>")
    out.append("<tbody>")
    for row in rows:
        out.append("<tr>")
        out.append(f"<td>{html.escape(str(row['mode']))}</td>")
        out.append(f"<td>{row['alive']}</td>")
        out.append(f"<td>{row['lost']}</td>")
        for coord, _unit in COORDS:
            out.append(f"<td>{fmt(float(row[f'rms_{coord}']))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def mean_delta_table(rows: list[dict[str, object]]) -> str:
    out = ["<table>", "<thead><tr><th>Mode</th><th>Mean delta at endpoint</th></tr></thead>", "<tbody>"]
    for row in rows:
        out.append(f"<tr><td>{html.escape(str(row['mode']))}</td><td>{fmt(float(row['mean_delta']))}</td></tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def main() -> None:
    rows = collect()
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FACET Xsuite Tracking SR Comparison</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 28px;
      color: #1f2937;
      line-height: 1.45;
    }}
    h1, h2 {{ color: #111827; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 14px 0 26px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 7px 9px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{
      max-width: 100%;
      border: 1px solid #d1d5db;
    }}
    .note {{ color: #4b5563; }}
    code {{ background: #f3f4f6; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>FACET Xsuite Particle Tracking: Synchrotron Radiation Comparison</h1>
  <p>
    Compared three one-pass CPU tracking runs using the same realistic independent x/y FACET input distribution:
    no synchrotron radiation, mean synchrotron radiation, and quantum synchrotron radiation.
    Each run used 10,000 macroparticles and saved the start/end particle distributions.
  </p>

  <h2>Endpoint RMS Values</h2>
  <p class="note">RMS values are computed over particles with positive final state. Scientific notation is kept to expose small differences between the SR modes.</p>
  {table(rows)}

  <h2>Endpoint Mean Energy Offset</h2>
  {mean_delta_table(rows)}

  <h2>RMS-Normalized Phase-Space Comparison</h2>
  <p class="note">The overlaid start/end plot uses coordinates normalized by each distribution's own RMS in that dimension. The physical RMS values are printed in each panel.</p>
  <img src="{html.escape(PLOT.name)}" alt="RMS-normalized phase-space comparison for no SR, mean SR, and quantum SR">

  <h2>Files</h2>
  <ul>
    <li><code>{html.escape(PLOT.name)}</code></li>
    <li><code>{html.escape(RUNS['No SR'].name)}</code></li>
    <li><code>{html.escape(RUNS['Mean SR'].name)}</code></li>
    <li><code>{html.escape(RUNS['Quantum SR'].name)}</code></li>
  </ul>
</body>
</html>
"""
    OUT_HTML.write_text(html_text, encoding="utf-8")
    print(f"wrote={OUT_HTML}")


if __name__ == "__main__":
    main()
