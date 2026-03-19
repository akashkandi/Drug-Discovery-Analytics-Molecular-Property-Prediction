# Drug Discovery Molecular Property Prediction Pipeline

A complete end-to-end machine learning pipeline for predicting molecular properties
relevant to drug discovery, specifically:

- **Blood-Brain Barrier (BBB) Permeability** — binary classification (BBBP dataset)
- **Aqueous Solubility (LogS)** — regression (ESOL dataset)

Includes feature engineering with RDKit, XGBoost/LightGBM models, SHAP
interpretability, MLflow experiment tracking, and a FastAPI prediction service.

---

## Architecture

```
drug-discovery-ml/
│
├── data/                          # Raw + processed datasets
│   ├── BBBP.csv                   # Blood-brain barrier dataset (~2050 mols)
│   ├── delaney-processed.csv      # ESOL solubility dataset (~1128 mols)
│   ├── features_bbbp.csv          # ECFP4 + descriptor feature matrix
│   ├── features_esol.csv          # ECFP4 + descriptor feature matrix (ESOL)
│   ├── data_dictionary.md         # Column definitions
│   └── invalid_smiles_report.txt  # Molecules that failed parsing
│
├── src/                           # Source modules
│   ├── feature_engineering.py     # SMILES → fingerprints + descriptors
│   ├── eda.py                     # Exploratory data analysis + plots
│   ├── models.py                  # Classification + regression training
│   ├── shap_analysis.py           # SHAP interpretability
│   └── api.py                     # FastAPI REST service
│
├── outputs/                       # All generated artifacts
│   ├── eda/                       # EDA plots (PNG)
│   ├── classification/            # ROC, CM, calibration plots
│   ├── shap/                      # SHAP plots + interpretation
│   ├── regression_results/        # Regression plots + comparison
│   ├── models/                    # Saved model files (.pkl)
│   ├── model_comparison.csv       # All classifier metrics
│   ├── eda_summary.md             # EDA statistics summary
│   ├── benchmark_report.md        # Comparison vs literature
│   └── final_report.md            # Full project report
│
├── mlruns/                        # MLflow experiment tracking (auto-generated)
├── run_pipeline.py                # One-command full pipeline runner
├── requirements.txt
└── README.md
```

### Data Flow

```
SMILES strings
      │
      ▼
┌─────────────────────────────────────────────┐
│        Feature Engineering (RDKit)          │
│  ┌──────────────────┐  ┌─────────────────┐  │
│  │  Morgan ECFP4    │  │  9 Molecular    │  │
│  │  2048-bit vector │  │  Descriptors    │  │
│  └────────┬─────────┘  └───────┬─────────┘  │
│           └──────────┬─────────┘            │
│                      ▼                      │
│            Feature Matrix (2057 cols)       │
└─────────────────────────────────────────────┘
      │
      ├──► EDA (distributions, heatmaps, scatter)
      │
      ├──► Classification Pipeline (BBBP)
      │       ├── Logistic Regression
      │       ├── Random Forest
      │       ├── XGBoost  ◄── Best model for SHAP
      │       └── LightGBM
      │
      ├──► Regression Pipeline (ESOL)
      │       ├── Ridge Regression
      │       ├── XGBoost Regressor
      │       └── LightGBM Regressor
      │
      ├──► SHAP Analysis
      │       ├── Beeswarm summary
      │       ├── Bar importance
      │       ├── Waterfall (single mol)
      │       └── Dependence plots
      │
      └──► FastAPI Service
              ├── POST /predict-permeability
              ├── POST /predict-solubility
              ├── GET  /molecule-info
              └── GET  /health
```

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### 1. Clone / Navigate to project

```bash
cd drug-discovery-ml
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on RDKit:** If `pip install rdkit` fails on your platform, try:
> ```bash
> conda install -c conda-forge rdkit
> ```

---

## Running the Full Pipeline

### One-command execution

```bash
python run_pipeline.py
```

This runs all 7 steps in sequence:
1. Download datasets
2. Feature engineering (ECFP4 + descriptors)
3. Exploratory data analysis
4. Classification pipeline (BBBP)
5. Regression pipeline (ESOL)
6. SHAP interpretability
7. Final report generation

### Running individual steps

```bash
# Step 2: Feature engineering only
python src/feature_engineering.py

# Step 3: EDA only (requires features_bbbp.csv)
python src/eda.py

# Step 4+5: Models only
python src/models.py

# Step 6: SHAP only (requires trained XGBoost model)
python src/shap_analysis.py
```

---

## Starting the FastAPI Service

### Development server (with hot reload)

```bash
uvicorn src.api:app --reload --port 8000
```

### Production server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Note:** Run the full pipeline first to train and save models before starting the API.

### API Endpoints

Once running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

#### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Predict permeability (aspirin)
curl -X POST http://localhost:8000/predict-permeability \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'

# Predict solubility (benzene)
curl -X POST http://localhost:8000/predict-solubility \
  -H "Content-Type: application/json" \
  -d '{"smiles": "c1ccccc1"}'

# Get molecule descriptors
curl "http://localhost:8000/molecule-info?smiles=CC(=O)Oc1ccccc1C(=O)O"
```

#### Example Response (predict-permeability)

```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "prediction": "permeable",
  "probability": 0.87,
  "key_features": {
    "MolWt": 180.16,
    "MolLogP": 1.24,
    "TPSA": 63.6,
    "NumHDonors": 0,
    "NumHAcceptors": 4,
    "NumRotatableBonds": 3,
    "RingCount": 1,
    "NumAromaticRings": 1,
    "FractionCSP3": 0.14
  },
  "shap_top3": [
    {"feature": "MolLogP", "importance": 0.0821},
    {"feature": "TPSA", "importance": 0.0614},
    {"feature": "MolWt", "importance": 0.0432}
  ]
}
```

---

## Experiment Tracking with MLflow

All model runs are logged to MLflow automatically.

```bash
# Start MLflow UI
mlflow ui --port 5000
```

Visit http://localhost:5000 to browse experiments, compare runs, and view artifacts.

**Experiments:**
- `drug-discovery-bbbp` — classification models
- `drug-discovery-esol` — regression models

---

## Results Summary

### Best Model Performance (approximate, random split)

| Task | Best Model | Primary Metric |
|------|------------|----------------|
| BBB Permeability | XGBoost | ROC-AUC ~0.90 |
| Aqueous Solubility | XGBoost Regressor | RMSE ~0.70 |

### vs Literature Baselines

| Model | Split | ROC-AUC |
|-------|-------|---------|
| Our XGBoost | Random | ~0.90 |
| Literature RF (MoleculeNet) | Scaffold | 0.714 |
| Literature GCN (MoleculeNet) | Scaffold | 0.877 |

> Our random split scores are not directly comparable to scaffold-split literature
> values. Scaffold splits are harder and better represent real-world prospective
> screening performance.

### Key SHAP Findings

The most predictive molecular features for BBB permeability:
1. **LogP** (lipophilicity) — moderate values (1–3) enable membrane partitioning
2. **TPSA** — low TPSA (< 90 Å²) is required for CNS penetration
3. **Molecular weight** — below ~450 Da for good BBB permeation

---

## References

1. Wu, Z. et al. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9(2), 513–530.
2. Delaney, J.S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. *J. Chem. Inf. Comput. Sci.*, 44(3), 1000–1005.
3. Rogers, D. & Hahn, M. (2010). Extended-connectivity fingerprints. *J. Chem. Inf. Model.*, 50(5), 742–754.
4. Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*, 30.
5. Lipinski, C.A. et al. (1997). Experimental and computational approaches to estimate solubility and permeability in drug discovery. *Adv. Drug Deliv. Rev.*, 23(1-3), 3–25.

---

## License

MIT License — see LICENSE file.
