"""Upload, clean, and visualize XFOIL polar data files.

Run with: streamlit run xfoil_polar_viewer.py
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


CANONICAL_COLUMNS = {
    "alpha": "Alpha (deg)",
    "aoa": "Alpha (deg)",
    "angleofattack": "Alpha (deg)",
    "cl": "CL",
    "liftcoefficient": "CL",
    "cd": "CD",
    "dragcoefficient": "CD",
    "cdp": "CDp",
    "cm": "CM",
    "pitchingmoment": "CM",
    "topxtr": "Top Xtr",
    "toptransition": "Top Xtr",
    "botxtr": "Bot Xtr",
    "bottomxtr": "Bot Xtr",
    "bottomtransition": "Bot Xtr",
}

EXPECTED_XFOIL_COLUMNS = ["Alpha (deg)", "CL", "CD", "CDp", "CM", "Top Xtr", "Bot Xtr"]


@dataclass
class PolarFile:
    name: str
    data: pd.DataFrame
    metadata: dict[str, str]
    skipped_rows: int = 0


def compact_name(value: object) -> str:
    """Make a header safe to compare despite spaces, underscores, or case."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: CANONICAL_COLUMNS.get(compact_name(column), str(column).strip())
        for column in frame.columns
    }
    result = frame.rename(columns=renamed).copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="ignore")
    return result


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add aerodynamic efficiency where both lift and drag are available."""
    result = frame.copy()
    if {"CL", "CD"}.issubset(result.columns):
        drag = pd.to_numeric(result["CD"], errors="coerce").where(lambda values: values > 0)
        lift = pd.to_numeric(result["CL"], errors="coerce")
        result["L/D"] = lift / drag
    return result


def extract_xfoil_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines:
        match = re.search(r"Calculated polar for:\s*(.+)", line, flags=re.I)
        if match:
            metadata["Airfoil"] = match.group(1).strip()
        match = re.search(r"Mach\s*=\s*([\d.]+)", line, flags=re.I)
        if match:
            metadata["Mach"] = match.group(1)
        match = re.search(r"Re\s*=\s*([\d.]+)\s*e\s*([+-]?\d+)", line, flags=re.I)
        if match:
            metadata["Reynolds number"] = f"{float(match.group(1)) * 10 ** int(match.group(2)):.3g}"
        match = re.search(r"Ncrit\s*=\s*([\d.]+)", line, flags=re.I)
        if match:
            metadata["Ncrit"] = match.group(1)
    return metadata


def parse_xfoil_text(text: str) -> tuple[pd.DataFrame, dict[str, str], int] | None:
    """Parse an XFOIL polar even when XFOIL has left incomplete rows."""
    lines = text.splitlines()
    header_row = next(
        (index for index, line in enumerate(lines) if re.search(r"\balpha\b", line, re.I) and re.search(r"\bCL\b", line, re.I)),
        None,
    )
    if header_row is None:
        return None

    raw_headers = re.findall(r"[A-Za-z][A-Za-z_]*", lines[header_row])
    headers = [CANONICAL_COLUMNS.get(compact_name(value), value) for value in raw_headers]
    if "Alpha (deg)" not in headers or "CL" not in headers:
        return None

    values: list[list[float]] = []
    skipped = 0
    numeric_pattern = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[Ee][-+]?\d+)?"
    for line in lines[header_row + 1 :]:
        numbers = re.findall(numeric_pattern, line)
        if not numbers:
            continue
        if len(numbers) < len(headers):
            skipped += 1
            continue
        values.append([float(number) for number in numbers[: len(headers)]])

    if not values:
        return None
    return pd.DataFrame(values, columns=headers), extract_xfoil_metadata(lines), skipped


def parse_delimited_text(text: str) -> pd.DataFrame:
    """Read CSV, TSV, or a simple whitespace-separated table."""
    try:
        frame = pd.read_csv(io.StringIO(text), sep=None, engine="python", comment="#")
    except (pd.errors.ParserError, UnicodeDecodeError):
        frame = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python", comment="#")
    return standardize_columns(frame).dropna(axis=1, how="all")


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_name: str, file_bytes: bytes) -> PolarFile:
    suffix = Path(file_name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = add_derived_metrics(standardize_columns(pd.read_excel(io.BytesIO(file_bytes))))
        return PolarFile(file_name, frame, {})

    text = file_bytes.decode("utf-8", errors="replace")
    parsed = parse_xfoil_text(text)
    if parsed:
        frame, metadata, skipped = parsed
        return PolarFile(file_name, add_derived_metrics(frame), metadata, skipped)
    return PolarFile(file_name, add_derived_metrics(parse_delimited_text(text)), {})


def chart_for(files: list[PolarFile], metric: str):
    long_data = []
    for item in files:
        subset = item.data[["Alpha (deg)", metric]].dropna().copy()
        subset["Airfoil / file"] = item.metadata.get("Airfoil", Path(item.name).stem)
        long_data.append(subset)
    combined = pd.concat(long_data, ignore_index=True)
    labels = {
        "CL": "Lift coefficient, CL",
        "CD": "Drag coefficient, CD",
        "CDp": "Pressure drag coefficient, CDp",
        "CM": "Pitching-moment coefficient, CM",
        "L/D": "Aerodynamic efficiency, L/D",
        "Top Xtr": "Upper-surface transition location, x/c",
        "Bot Xtr": "Lower-surface transition location, x/c",
    }
    figure = px.line(
        combined,
        x="Alpha (deg)",
        y=metric,
        color="Airfoil / file",
        markers=True,
        labels={"Alpha (deg)": "Angle of attack, α (deg)", metric: labels.get(metric, metric)},
        template="plotly_white",
    )
    figure.update_layout(legend_title_text="", margin=dict(l=20, r=20, t=25, b=20), hovermode="x unified")
    figure.update_traces(line=dict(width=2))
    return figure


st.set_page_config(page_title="XFOIL Polar Viewer", page_icon="✈️", layout="wide")
st.title("XFOIL Polar Viewer")
st.caption("Upload XFOIL .text/.txt, CSV/TSV, or Excel files to plot airfoil polar data.")

uploads = st.file_uploader(
    "Upload one or more polar data files",
    type=["text", "txt", "csv", "tsv", "xlsx", "xls"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("Choose your NACA_0012.text file to begin. You can add NACA_2412 and NACA_4412 files for comparison.")
    st.stop()

polars: list[PolarFile] = []
for upload in uploads:
    try:
        polar = read_uploaded_file(upload.name, upload.getvalue())
        if "Alpha (deg)" not in polar.data.columns:
            st.warning(f"{upload.name}: no alpha/angle-of-attack column was found, so this file was skipped.")
            continue
        polars.append(polar)
    except Exception as error:  # Keeps one bad upload from stopping the remaining files.
        st.error(f"Could not read {upload.name}: {error}")

if not polars:
    st.stop()

available_metrics = [
    metric for metric in ["CL", "CD", "CDp", "CM", "L/D", "Top Xtr", "Bot Xtr"] if any(metric in polar.data.columns for polar in polars)
]
metric = st.selectbox("Plot", available_metrics, format_func=lambda value: {
    "CL": "Lift coefficient (CL)", "CD": "Drag coefficient (CD)", "CDp": "Pressure drag (CDp)",
    "CM": "Pitching moment (CM)", "L/D": "Aerodynamic efficiency (L/D)", "Top Xtr": "Top transition location", "Bot Xtr": "Bottom transition location",
}.get(value, value))

valid_for_metric = [polar for polar in polars if metric in polar.data.columns]
st.plotly_chart(chart_for(valid_for_metric, metric), use_container_width=True)

with st.expander("File details and cleaned data"):
    for polar in polars:
        st.subheader(polar.metadata.get("Airfoil", Path(polar.name).stem))
        if polar.metadata:
            st.write(" · ".join(f"{key}: {value}" for key, value in polar.metadata.items()))
        if polar.skipped_rows:
            st.warning(f"Skipped {polar.skipped_rows} incomplete XFOIL data row(s).")
        st.dataframe(polar.data, use_container_width=True, hide_index=True)
        st.download_button(
            f"Download cleaned CSV — {polar.name}",
            polar.data.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(polar.name).stem}_cleaned.csv",
            mime="text/csv",
        )

