import logging
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Metrics present in every result.csv
METRICS = ["obj_iou", "io_ratio", "wdp"]

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def _extract_numeric_param(param_str: str) -> float:
    matches = _NUM_RE.findall(param_str)
    if not matches:
        return 0.0
    return float(matches[-1])


def _parse_csv_path(csv_path: Path, output_root: Path) -> Optional[dict]:
    try:
        rel = csv_path.relative_to(output_root)
        parts = rel.parts  # e.g. ("AdversarialPatch", "patch_ratio_0.07", "650", "result.csv")
    except ValueError:
        return None

    # parts[-1] is always "result.csv"
    if len(parts) == 4:
        perturbation_type, param_str, img_id, _ = parts
    elif len(parts) == 3:
        # Baseline layout: no param sub-folder
        perturbation_type, img_id, _ = parts
        param_str = "Normale"
    else:
        logger.debug("Unexpected path depth (%d parts), skipping: %s", len(parts), csv_path)
        return None

    return {
        "perturbation_type": perturbation_type,
        "perturbation_param": param_str,
        "param_value": _extract_numeric_param(param_str),
        "img_id": img_id,
    }

def load_results(output_dir: str = "output") -> pd.DataFrame:
    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Output directory not found: {root.resolve()}")

    records = []
    csv_files = sorted(root.rglob("results.csv"))

    if not csv_files:
        logger.warning("No results.csv files found under: %s", root.resolve())
        return pd.DataFrame(columns=["perturbation_type", "perturbation_param",
                                     "param_value", "img_id", "category"] + METRICS)

    for csv_path in csv_files:
        meta = _parse_csv_path(csv_path, root)
        if meta is None:
            continue

        try:
            df_csv = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", csv_path, exc)
            continue

        if df_csv.empty:
            continue

        # Rename readable_token -> category for consistency
        if "readable_token" in df_csv.columns:
            df_csv = df_csv.rename(columns={"readable_token": "category"})

        # Keep only the columns we need
        keep = ["category"] + [m for m in METRICS if m in df_csv.columns]
        df_csv = df_csv[keep].copy()

        for col, val in meta.items():
            df_csv[col] = val

        records.append(df_csv)

    if not records:
        logger.warning("All CSV files were empty or unparseable.")
        return pd.DataFrame()

    df = pd.concat(records, ignore_index=True)

    # Reorder columns for readability
    col_order = ["perturbation_type", "perturbation_param", "param_value",
                 "img_id", "category"] + METRICS
    df = df[[c for c in col_order if c in df.columns]]

    logger.info(
        "Loaded %d rows from %d CSV files across %d perturbation type(s).",
        len(df), len(csv_files),
        df["perturbation_type"].nunique() if not df.empty else 0,
    )
    return df