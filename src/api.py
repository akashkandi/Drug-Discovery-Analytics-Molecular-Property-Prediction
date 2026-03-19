"""
FastAPI Drug Discovery Service
================================
Provides REST endpoints for:
  - POST /predict-permeability  — BBB permeability prediction
  - POST /predict-solubility    — aqueous solubility (LogS) prediction
  - GET  /molecule-info         — all molecular descriptors from SMILES
  - GET  /health                — service health check

Usage
-----
    uvicorn src.api:app --reload --port 8000
"""

import os
import json
import warnings
from typing import Optional, List, Dict, Any

import numpy as np
import joblib

warnings.filterwarnings("ignore")

# ── FastAPI ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── RDKit ──────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ── SHAP ───────────────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
MORGAN_RADIUS = 2
MORGAN_NBITS  = 2048

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3",
]

# Paths to trained models (resolved relative to this file or cwd)
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_BASE, "outputs", "models")

CLASSIF_MODEL_PATHS = [
    os.path.join(MODELS_DIR, "xgboost.pkl"),
    os.path.join(MODELS_DIR, "lightgbm.pkl"),
    os.path.join(MODELS_DIR, "random_forest.pkl"),
]
REGRESS_MODEL_PATHS = [
    os.path.join(MODELS_DIR, "xgboost_regressor_reg.pkl"),
    os.path.join(MODELS_DIR, "lightgbm_regressor_reg.pkl"),
    os.path.join(MODELS_DIR, "ridge_regression_reg.pkl"),
]
SHAP_IMPORTANCE_PATH = os.path.join(_BASE, "outputs", "shap", "shap_importance.json")


# ── Model Loading (lazy, cached at module level) ───────────────────────────────

_classif_model = None
_regress_model = None
_shap_explainer = None
_shap_importance: Dict[str, float] = {}
_feature_names: List[str] = []


def _find_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _load_classif_model():
    global _classif_model
    if _classif_model is not None:
        return _classif_model
    path = _find_first(CLASSIF_MODEL_PATHS)
    if path is None:
        return None
    _classif_model = joblib.load(path)
    return _classif_model


def _load_regress_model():
    global _regress_model
    if _regress_model is not None:
        return _regress_model
    path = _find_first(REGRESS_MODEL_PATHS)
    if path is None:
        return None
    _regress_model = joblib.load(path)
    return _regress_model


def _load_shap_importance() -> Dict[str, float]:
    global _shap_importance
    if _shap_importance:
        return _shap_importance
    if os.path.exists(SHAP_IMPORTANCE_PATH):
        with open(SHAP_IMPORTANCE_PATH) as f:
            _shap_importance = json.load(f)
    return _shap_importance


def _build_feature_names() -> List[str]:
    """Return ordered feature names (fp_0..fp_2047 + descriptors)."""
    global _feature_names
    if _feature_names:
        return _feature_names
    _feature_names = [f"fp_{i}" for i in range(MORGAN_NBITS)] + DESCRIPTOR_NAMES
    return _feature_names


# ── Molecular Feature Computation ─────────────────────────────────────────────

def _compute_descriptors(mol) -> Optional[Dict[str, float]]:
    if mol is None:
        return None
    try:
        return {
            "MolWt":             round(Descriptors.MolWt(mol), 4),
            "MolLogP":           round(Descriptors.MolLogP(mol), 4),
            "TPSA":              round(Descriptors.TPSA(mol), 4),
            "NumHDonors":        int(rdMolDescriptors.CalcNumHBD(mol)),
            "NumHAcceptors":     int(rdMolDescriptors.CalcNumHBA(mol)),
            "NumRotatableBonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "RingCount":         int(rdMolDescriptors.CalcNumRings(mol)),
            "NumAromaticRings":  int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "FractionCSP3":      round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 4),
        }
    except Exception:
        return None


def _smiles_to_feature_vector(smiles: str) -> Optional[np.ndarray]:
    """
    Convert SMILES → flat numpy array [fp_0..fp_2047, desc_0..desc_8].
    Returns None if SMILES is invalid.
    """
    if not RDKIT_AVAILABLE:
        raise HTTPException(status_code=503, detail="RDKit not available on server.")

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    fp_arr = np.array(fp, dtype=np.float32)

    desc = _compute_descriptors(mol)
    if desc is None:
        return None

    desc_arr = np.array([desc[k] for k in DESCRIPTOR_NAMES], dtype=np.float32)
    return np.concatenate([fp_arr, desc_arr]).reshape(1, -1)


# ── Pydantic Schemas ───────────────────────────────────────────────────────────

class PermeabilityRequest(BaseModel):
    smiles: str = Field(..., example="CC(=O)Oc1ccccc1C(=O)O",
                        description="SMILES string of the molecule")

class PermeabilityResponse(BaseModel):
    smiles:      str
    prediction:  str    # "permeable" | "non-permeable"
    probability: float
    key_features: Dict[str, float]
    shap_top3:   List[Dict[str, Any]]

class SolubilityRequest(BaseModel):
    smiles: str = Field(..., example="c1ccccc1",
                        description="SMILES string of the molecule")

class SolubilityResponse(BaseModel):
    smiles:          str
    predicted_logS:  float
    confidence:      str   # "high" | "medium" | "low"

class MoleculeInfoResponse(BaseModel):
    smiles:      str
    descriptors: Dict[str, float]
    valid:       bool

class HealthResponse(BaseModel):
    status:            str
    rdkit_available:   bool
    classif_model_loaded: bool
    regress_model_loaded: bool
    shap_available:    bool


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Drug Discovery Molecular Property Prediction API",
    description=(
        "Predicts blood-brain barrier permeability and aqueous solubility "
        "from SMILES strings using trained ML models (XGBoost / LightGBM)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
async def startup_event():
    """Pre-load models at startup for faster first-request latency."""
    print("[API] Loading models...")
    _load_classif_model()
    _load_regress_model()
    _load_shap_importance()
    _build_feature_names()
    print("[API] Ready.")


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    """Service health check — returns status and model availability."""
    return HealthResponse(
        status="ok",
        rdkit_available=RDKIT_AVAILABLE,
        classif_model_loaded=_classif_model is not None or _find_first(CLASSIF_MODEL_PATHS) is not None,
        regress_model_loaded=_regress_model is not None or _find_first(REGRESS_MODEL_PATHS) is not None,
        shap_available=SHAP_AVAILABLE,
    )


@app.post("/predict-permeability", response_model=PermeabilityResponse,
          tags=["Prediction"])
def predict_permeability(req: PermeabilityRequest):
    """
    Predict blood-brain barrier (BBB) permeability from a SMILES string.

    Returns
    -------
    - **prediction**: "permeable" or "non-permeable"
    - **probability**: probability of being permeable (0–1)
    - **key_features**: calculated molecular descriptors
    - **shap_top3**: top 3 SHAP-important features with their importance values
    """
    model = _load_classif_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Classification model not loaded. Run the training pipeline first.",
        )

    X = _smiles_to_feature_vector(req.smiles)
    if X is None:
        raise HTTPException(status_code=422,
                            detail=f"Invalid or unparseable SMILES: '{req.smiles}'")

    prob = float(model.predict_proba(X)[0, 1])
    label = "permeable" if prob >= 0.5 else "non-permeable"

    # Key descriptors
    mol = Chem.MolFromSmiles(req.smiles.strip())
    key_features = _compute_descriptors(mol) or {}

    # SHAP top3 from pre-computed importance (fast, no re-computation)
    importance = _load_shap_importance()
    shap_top3 = [
        {"feature": k, "importance": round(v, 4)}
        for k, v in list(importance.items())[:3]
    ]

    return PermeabilityResponse(
        smiles=req.smiles,
        prediction=label,
        probability=round(prob, 4),
        key_features={k: round(float(v), 4) for k, v in key_features.items()},
        shap_top3=shap_top3,
    )


@app.post("/predict-solubility", response_model=SolubilityResponse,
          tags=["Prediction"])
def predict_solubility(req: SolubilityRequest):
    """
    Predict aqueous solubility (LogS) from a SMILES string.

    Returns
    -------
    - **predicted_logS**: log10 molar solubility prediction
    - **confidence**: "high" (|LogS| < 1), "medium", or "low"
    """
    model = _load_regress_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Regression model not loaded. Run the training pipeline first.",
        )

    X = _smiles_to_feature_vector(req.smiles)
    if X is None:
        raise HTTPException(status_code=422,
                            detail=f"Invalid or unparseable SMILES: '{req.smiles}'")

    log_s = float(model.predict(X)[0])

    # Simple heuristic confidence from absolute value range
    if -4 <= log_s <= 0:
        confidence = "high"
    elif -7 <= log_s <= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return SolubilityResponse(
        smiles=req.smiles,
        predicted_logS=round(log_s, 4),
        confidence=confidence,
    )


@app.get("/molecule-info", response_model=MoleculeInfoResponse,
         tags=["Utility"])
def molecule_info(smiles: str = Query(..., description="SMILES string of the molecule",
                                       example="CC(=O)Oc1ccccc1C(=O)O")):
    """
    Calculate and return all molecular descriptors for a given SMILES.

    Returns RDKit-computed physicochemical properties without running
    the prediction model.
    """
    if not RDKIT_AVAILABLE:
        raise HTTPException(status_code=503, detail="RDKit not available.")

    mol = Chem.MolFromSmiles(smiles.strip()) if smiles else None
    if mol is None:
        return MoleculeInfoResponse(smiles=smiles, descriptors={}, valid=False)

    desc = _compute_descriptors(mol) or {}
    return MoleculeInfoResponse(
        smiles=smiles,
        descriptors={k: round(float(v), 4) for k, v in desc.items()},
        valid=True,
    )


# ── Run locally ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
