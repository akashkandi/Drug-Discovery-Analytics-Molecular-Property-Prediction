"""
Feature Engineering Module for Drug Discovery Pipeline
=======================================================
Generates Morgan fingerprints (ECFP4) and RDKit molecular descriptors
from SMILES strings. Handles invalid SMILES gracefully.
"""

import os
import warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict

warnings.filterwarnings("ignore")

# RDKit imports
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")  # Suppress RDKit warnings
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[ERROR] RDKit not available. Install with: pip install rdkit")

# ── Constants ────────────────────────────────────────────────────────────────
BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
]


# ── Data Download ─────────────────────────────────────────────────────────────

def download_dataset(url: str, save_path: str, force: bool = False) -> str:
    """
    Download a dataset CSV from a URL and save to disk.

    Parameters
    ----------
    url : str
        URL to download from.
    save_path : str
        Local file path to save the CSV.
    force : bool
        Re-download even if file already exists.

    Returns
    -------
    str
        Path to the saved file.
    """
    if os.path.exists(save_path) and not force:
        print(f"  [SKIP] File already exists: {save_path}")
        return save_path

    print(f"  [DOWNLOAD] {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(response.content)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"  [OK] Saved {size_kb:.1f} KB -> {save_path}")
    return save_path


# ── Molecular Descriptor Computation ─────────────────────────────────────────

def compute_descriptors(mol) -> Optional[Dict[str, float]]:
    """
    Compute a fixed set of RDKit molecular descriptors for a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object (already parsed).

    Returns
    -------
    dict or None
        Dictionary of descriptor name → value, or None if computation fails.
    """
    if mol is None:
        return None
    try:
        return {
            "MolWt":             Descriptors.MolWt(mol),
            "MolLogP":           Descriptors.MolLogP(mol),
            "TPSA":              Descriptors.TPSA(mol),
            "NumHDonors":        rdMolDescriptors.CalcNumHBD(mol),
            "NumHAcceptors":     rdMolDescriptors.CalcNumHBA(mol),
            "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "RingCount":         rdMolDescriptors.CalcNumRings(mol),
            "NumAromaticRings":  rdMolDescriptors.CalcNumAromaticRings(mol),
            "FractionCSP3":      rdMolDescriptors.CalcFractionCSP3(mol),
        }
    except Exception as exc:
        print(f"  [WARN] Descriptor computation failed: {exc}")
        return None


def compute_morgan_fingerprint(mol, radius: int = MORGAN_RADIUS,
                                n_bits: int = MORGAN_NBITS) -> Optional[np.ndarray]:
    """
    Compute Morgan (ECFP4) fingerprint as a binary bit vector.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
    radius : int
        Morgan radius (2 = ECFP4).
    n_bits : int
        Number of bits in the fingerprint vector.

    Returns
    -------
    numpy.ndarray or None
        Binary array of length n_bits.
    """
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception as exc:
        print(f"  [WARN] Fingerprint computation failed: {exc}")
        return None


def smiles_to_features(smiles: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Convert a SMILES string into a (fingerprint, descriptors) pair.

    Parameters
    ----------
    smiles : str
        SMILES string for a molecule.

    Returns
    -------
    tuple (fingerprint_array, descriptor_dict)
        Both are None if the SMILES is invalid.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None, None

    fp = compute_morgan_fingerprint(mol)
    desc = compute_descriptors(mol)
    return fp, desc


# ── Dataset Feature Matrix Construction ──────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame,
                          smiles_col: str = "smiles",
                          label_col: Optional[str] = None,
                          dataset_name: str = "dataset") -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a full feature matrix from a DataFrame containing SMILES strings.

    Combines Morgan fingerprint bits (fp_0 … fp_2047) with molecular
    descriptors into a single wide DataFrame.  Invalid SMILES are skipped
    and reported.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least a SMILES column.
    smiles_col : str
        Name of the SMILES column.
    label_col : str or None
        Optional label column to carry over into the feature matrix.
    dataset_name : str
        Name used in progress messages.

    Returns
    -------
    (feature_df, invalid_smiles_list)
        feature_df : pd.DataFrame with columns [fp_0..fp_2047, descriptors, label?]
        invalid_smiles_list : list of invalid/failed SMILES strings
    """
    print(f"\n[FEATURE ENG] Building feature matrix for {dataset_name} ({len(df)} rows)...")

    fp_cols = [f"fp_{i}" for i in range(MORGAN_NBITS)]
    records = []
    invalid_smiles = []

    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        fp, desc = smiles_to_features(smiles)

        if fp is None or desc is None:
            invalid_smiles.append(smiles)
            continue

        record = {}
        # Fingerprint bits
        for i, bit in enumerate(fp):
            record[f"fp_{i}"] = int(bit)
        # Molecular descriptors
        record.update(desc)
        # Label (if present)
        if label_col and label_col in df.columns:
            record[label_col] = row[label_col]
        # Carry metadata
        if "name" in df.columns:
            record["name"] = row["name"]
        record["smiles"] = smiles

        records.append(record)

    feature_df = pd.DataFrame(records)

    print(f"  [OK] {len(feature_df)} valid molecules | {len(invalid_smiles)} invalid SMILES")
    if invalid_smiles:
        print(f"  [WARN] Invalid SMILES ({len(invalid_smiles)}):")
        for s in invalid_smiles[:10]:
            print(f"    - {s}")
        if len(invalid_smiles) > 10:
            print(f"    ... and {len(invalid_smiles) - 10} more")

    return feature_df, invalid_smiles


# ── ESOL-specific helpers ─────────────────────────────────────────────────────

def load_esol(csv_path: str) -> pd.DataFrame:
    """
    Load and normalise the ESOL (Delaney) solubility dataset.

    The column containing experimental LogS values may vary slightly
    across versions; this function tries several known column names.

    Parameters
    ----------
    csv_path : str
        Path to delaney-processed.csv

    Returns
    -------
    pd.DataFrame with at least columns: smiles, logS
    """
    df = pd.read_csv(csv_path)
    print(f"  [LOAD] ESOL columns: {df.columns.tolist()}")

    # Normalise SMILES column name
    smiles_candidates = ["smiles", "SMILES", "Smiles"]
    for c in smiles_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "smiles"})
            break

    # Normalise LogS column name
    logs_candidates = [
        "measured log solubility in mols per litre",
        "logS",
        "log_solubility",
        "Measured Log Solubility in Mols per Litre",
    ]
    for c in logs_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "logS"})
            break

    if "logS" not in df.columns:
        # Fall back: pick the most likely numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df = df.rename(columns={numeric_cols[0]: "logS"})
            print(f"  [WARN] Used '{numeric_cols[0]}' as logS column")

    return df[["smiles", "logS"]].dropna()


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_feature_engineering(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Full feature engineering pipeline:
      1. Download BBBP and ESOL datasets.
      2. Compute Morgan fingerprints + descriptors.
      3. Save feature matrices to CSV.

    Parameters
    ----------
    data_dir : str
        Directory for raw data and output feature CSVs.

    Returns
    -------
    dict with keys 'bbbp' and 'esol', each a feature DataFrame.
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for feature engineering.")

    os.makedirs(data_dir, exist_ok=True)

    # ── Download datasets ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Downloading datasets")
    print("="*60)

    bbbp_path = os.path.join(data_dir, "BBBP.csv")
    esol_path = os.path.join(data_dir, "delaney-processed.csv")
    download_dataset(BBBP_URL, bbbp_path)
    download_dataset(ESOL_URL, esol_path)

    # ── BBBP feature matrix ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — BBBP Feature Engineering")
    print("="*60)

    bbbp_raw = pd.read_csv(bbbp_path)
    print(f"  [LOAD] BBBP shape: {bbbp_raw.shape}")
    print(f"  [INFO] Class balance: {bbbp_raw['p_np'].value_counts().to_dict()}")

    bbbp_features, bbbp_invalid = build_feature_matrix(
        bbbp_raw,
        smiles_col="smiles",
        label_col="p_np",
        dataset_name="BBBP",
    )

    bbbp_out = os.path.join(data_dir, "features_bbbp.csv")
    bbbp_features.to_csv(bbbp_out, index=False)
    print(f"  [SAVED] {bbbp_out}  shape={bbbp_features.shape}")

    # ── ESOL feature matrix ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — ESOL Feature Engineering")
    print("="*60)

    esol_raw = load_esol(esol_path)
    print(f"  [LOAD] ESOL shape: {esol_raw.shape}")

    esol_features, esol_invalid = build_feature_matrix(
        esol_raw,
        smiles_col="smiles",
        label_col="logS",
        dataset_name="ESOL",
    )

    esol_out = os.path.join(data_dir, "features_esol.csv")
    esol_features.to_csv(esol_out, index=False)
    print(f"  [SAVED] {esol_out}  shape={esol_features.shape}")

    # ── Invalid SMILES report ─────────────────────────────────────────────────
    invalid_path = os.path.join(data_dir, "invalid_smiles_report.txt")
    with open(invalid_path, "w") as f:
        f.write("=== Invalid SMILES Report ===\n\n")
        f.write(f"BBBP invalid ({len(bbbp_invalid)}):\n")
        for s in bbbp_invalid:
            f.write(f"  {s}\n")
        f.write(f"\nESOL invalid ({len(esol_invalid)}):\n")
        for s in esol_invalid:
            f.write(f"  {s}\n")
    print(f"\n  [REPORT] Invalid SMILES saved → {invalid_path}")

    print("\n[FEATURE ENG] Complete.")
    return {"bbbp": bbbp_features, "esol": esol_features}


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run_feature_engineering(data_dir)
