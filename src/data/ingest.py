"""
Ingest Raman spectra from public datasets into a unified raw-standardized format.

Each spectrum is saved as an .npz with keys:
    wavenumber : float32 array, shape (N,), sorted ascending
    intensity  : float32 array, shape (N,), raw intensity (no L2 yet)
    chemical   : str, canonical chemical/sample name
    source     : str, dataset identifier
    kind       : "pure" | "mixture"
    meta       : JSON-serialized dict of source-specific extras

A row per spectrum is appended to data/manifest.csv.

Downstream: preprocess.py handles interpolation to a common grid + L2 normalization.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_STD_DIR = PROJECT_ROOT / "data" / "raw_std"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"

MANIFEST_COLUMNS = [
    "spectrum_id", "chemical", "source", "kind",
    "n_points", "wn_min", "wn_max", "wn_step",
    "npz_path", "meta_json",
]


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return s.lower() or "unknown"


def _save_spectrum(
    chemical: str,
    source: str,
    kind: str,
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    meta: dict,
    index: int,
) -> dict:
    """Save a single spectrum to raw_std and return a manifest row."""
    assert wavenumber.shape == intensity.shape
    assert wavenumber.ndim == 1
    order = np.argsort(wavenumber)
    wavenumber = wavenumber[order].astype(np.float32)
    intensity = intensity[order].astype(np.float32)

    spectrum_id = f"{source}__{_slugify(chemical)}__{index:05d}"
    out_dir = RAW_STD_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{spectrum_id}.npz"

    np.savez_compressed(
        npz_path,
        wavenumber=wavenumber,
        intensity=intensity,
        chemical=chemical,
        source=source,
        kind=kind,
        meta=json.dumps(meta),
    )

    diffs = np.diff(wavenumber)
    step = float(np.median(diffs)) if len(diffs) else 0.0

    return {
        "spectrum_id": spectrum_id,
        "chemical": chemical,
        "source": source,
        "kind": kind,
        "n_points": int(wavenumber.size),
        "wn_min": float(wavenumber.min()),
        "wn_max": float(wavenumber.max()),
        "wn_step": step,
        "npz_path": str(npz_path.relative_to(PROJECT_ROOT)),
        "meta_json": json.dumps(meta),
    }


# -----------------------------------------------------------------------------
# Source: API-Compound Raman library (Figshare 27931131)
# -----------------------------------------------------------------------------

def ingest_api_compound() -> Iterator[dict]:
    csv_path = RAW_DIR / "api_compound" / "raman_spectra_api_compounds.csv"
    df = pd.read_csv(csv_path)
    wn_cols = [c for c in df.columns if c != "label"]
    wavenumber = np.array([float(c) for c in wn_cols], dtype=np.float32)

    for i, row in df.iterrows():
        chemical = str(row["label"])
        intensity = row[wn_cols].to_numpy(dtype=np.float32)
        meta = {"row_index": int(i)}
        yield _save_spectrum(
            chemical=chemical, source="api_compound", kind="pure",
            wavenumber=wavenumber, intensity=intensity, meta=meta, index=i,
        )


# -----------------------------------------------------------------------------
# Source: RamanBioLib (GitHub mteranm/ramanbiolib)
# -----------------------------------------------------------------------------

def ingest_ramanbiolib() -> Iterator[dict]:
    db_dir = RAW_DIR / "ramanbiolib" / "ramanbiolib" / "db"
    spectra = pd.read_csv(db_dir / "raman_spectra_db.csv")
    metadata = pd.read_csv(db_dir / "metadata_db.csv")
    meta_by_id = {row["id"]: row.to_dict() for _, row in metadata.iterrows()}

    for i, row in spectra.iterrows():
        chemical = str(row["component"])
        wn = np.array(ast.literal_eval(row["wavenumbers"]), dtype=np.float32)
        intensity = np.array(ast.literal_eval(row["intensity"]), dtype=np.float32)
        meta = {"component_id": int(row["id"])}
        m = meta_by_id.get(row["id"], {})
        for k in ("type", "reference", "laser_wavelength", "raman_technique"):
            if k in m and pd.notna(m[k]):
                meta[k] = str(m[k])
        yield _save_spectrum(
            chemical=chemical, source="ramanbiolib", kind="pure",
            wavenumber=wn, intensity=intensity, meta=meta, index=i,
        )


# -----------------------------------------------------------------------------
# Source: Olive oil FTIR/Raman (Zenodo 14651816)
# -----------------------------------------------------------------------------

# Sample IDs look like "OLI0OLI001" or "OLI50SUN005" — parse: OLI{X}{OTHER}{rep}
# Convention: two 3-letter oil codes with a percentage between (or 0 at start = pure).
_OIL_ID_RE = re.compile(r"^([A-Z]{3})(\d+)([A-Z]{3})(\d+)$")


def _parse_oil_id(sid: str) -> tuple[str, dict]:
    """Return (canonical chemical name, meta dict) for a sample ID."""
    m = _OIL_ID_RE.match(sid.strip())
    if not m:
        return sid, {"sample_id": sid, "parse": "unrecognized"}
    a_code, a_pct, b_code, _rep = m.groups()
    a_pct = int(a_pct)
    if a_pct == 0:
        # Pure sample of b_code
        return f"oil:{b_code}", {"sample_id": sid, "kind": "pure", "component": b_code}
    # Mixture
    name = f"mix:{b_code}{100 - a_pct}_{a_code}{a_pct}"
    meta = {
        "sample_id": sid, "kind": "mixture",
        "components": {b_code: 100 - a_pct, a_code: a_pct},
    }
    return name, meta


def ingest_olive_oil() -> Iterator[dict]:
    raman_dir = RAW_DIR / "olive_oil" / "Data_set_FTIR_Raman" / "Raman"
    idx = 0
    for xlsx in sorted(raman_dir.glob("*.xlsx")):
        df = pd.read_excel(xlsx, sheet_name=0)
        id_col = df.columns[0]
        wn_cols = df.columns[1:]
        wavenumber = np.array([float(c) for c in wn_cols], dtype=np.float32)
        study = xlsx.stem.replace("Raman_spectra_", "")
        for _, row in df.iterrows():
            sid = str(row[id_col])
            chemical, meta = _parse_oil_id(sid)
            meta["study"] = study
            kind = meta.get("kind", "pure")
            intensity = row[wn_cols].to_numpy(dtype=np.float32)
            yield _save_spectrum(
                chemical=chemical, source="olive_oil", kind=kind,
                wavenumber=wavenumber, intensity=intensity, meta=meta, index=idx,
            )
            idx += 1


# -----------------------------------------------------------------------------
# Source: Sugar Mixtures (Zenodo 10779223, Georgiev et al. PNAS 2024)
#
# 245 wells (5 pure + 240 mixtures) of 4 sugars in water. Each well measured
# 8x at high SNR (Sugar_Concentration_Test) and 32x at low SNR (…_Fast).
# Ground-truth: sugar volumes in uL; we convert to volume fractions summing to 1.
# -----------------------------------------------------------------------------

SUGAR_COMPONENTS = ["Sucrose", "Fructose", "Maltose", "Glucose", "Water"]


def _sugar_composition(row: pd.Series) -> tuple[str, dict]:
    """Return (chemical name, meta with fractions) for a sugar-mixture measurement row."""
    total = float(row["Total Volume [ul]"])
    fracs = {c.lower(): float(row[f"{c} [ul]"]) / total for c in SUGAR_COMPONENTS}
    nonzero = {k: v for k, v in fracs.items() if v > 0}
    if len(nonzero) == 1:
        chemical = next(iter(nonzero))
        kind = "pure"
    else:
        chemical = "sugar_mix"
        kind = "mixture"
    meta = {"fractions": fracs, "kind": kind}
    return chemical, meta


def _ingest_sugar_csv(spectra_csv: Path, meta_csv: Path, snr_label: str, start_idx: int) -> Iterator[dict]:
    """Yield spectra from one of the two consolidated CSVs (high or low SNR)."""
    spectra = pd.read_csv(spectra_csv)
    meta_df = pd.read_csv(meta_csv)
    meta_by_filename = {row["filename"]: row for _, row in meta_df.iterrows()}

    wavenumber = spectra["Wavenumber [cm-1]"].to_numpy(dtype=np.float32)
    idx = start_idx
    for col in spectra.columns[1:]:
        m = meta_by_filename.get(col)
        if m is None:
            continue
        chemical, comp_meta = _sugar_composition(m)
        intensity = spectra[col].to_numpy(dtype=np.float32)
        meta = {
            **comp_meta,
            "snr": snr_label,
            "well": f"{m['row']}{m['col']}_{m['plate']}",
            "round": int(m["round"]),
            "rep": int(m["rep"]),
            "filename": col,
        }
        yield _save_spectrum(
            chemical=chemical, source="sugar_mixtures", kind=comp_meta["kind"],
            wavenumber=wavenumber, intensity=intensity, meta=meta, index=idx,
        )
        idx += 1


def ingest_sugar_mixtures() -> Iterator[dict]:
    base = RAW_DIR / "sugar_mixtures" / "Raw data" / "Experimental data from sugar mixtures" / "Raw data files"
    idx = 0
    # High SNR (8 reps per well = 1960 spectra)
    for row in _ingest_sugar_csv(
        base / "Sugar_Concentration_Test_ALL_spectra.csv",
        base / "Sugar_Concentration_Test_ALL_metadata.csv",
        snr_label="high", start_idx=idx,
    ):
        yield row
        idx += 1
    # Low SNR (32 reps per well = 7840 spectra)
    for row in _ingest_sugar_csv(
        base / "Sugar_Concentration_Test_Fast_ALL_spectra.csv",
        base / "Sugar_Concentration_Test_Fast_ALL_metadata.csv",
        snr_label="low", start_idx=idx,
    ):
        yield row
        idx += 1


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

SOURCES = {
    "api_compound": ingest_api_compound,
    "ramanbiolib": ingest_ramanbiolib,
    "olive_oil": ingest_olive_oil,
    "sugar_mixtures": ingest_sugar_mixtures,
}


def run_all(only: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for name, fn in SOURCES.items():
        if only and name not in only:
            continue
        print(f"[ingest] {name}: starting")
        n = 0
        for row in fn():
            rows.append(row)
            n += 1
            if n % 200 == 0:
                print(f"[ingest] {name}: {n} spectra")
        print(f"[ingest] {name}: done, {n} spectra")

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    if MANIFEST_PATH.exists():
        prior = pd.read_csv(MANIFEST_PATH)
        keep = prior[~prior["source"].isin(df["source"].unique())]
        df = pd.concat([keep, df], ignore_index=True)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"[ingest] wrote manifest {MANIFEST_PATH} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset of sources to ingest, e.g. --only api_compound ramanbiolib")
    args = ap.parse_args()
    run_all(only=args.only)
