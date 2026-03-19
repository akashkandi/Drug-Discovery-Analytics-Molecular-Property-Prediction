# Drug Discovery Molecular Property Prediction — Final Report

**Generated:** 2026-03-18
**Pipeline Version:** 1.0.0
**Author:** Drug Discovery ML Pipeline

---

## Executive Summary

This report documents a complete machine learning pipeline for predicting two
critical drug-likeness properties from molecular structure:

1. **Blood-Brain Barrier (BBB) Permeability** — binary classification on the
   BBBP dataset (MoleculeNet), indicating whether a molecule can cross from
   the bloodstream into the brain.

2. **Aqueous Solubility (LogS)** — regression on the ESOL (Delaney) dataset,
   predicting log10 molar solubility in water.

The pipeline uses ECFP4 Morgan fingerprints and 9 RDKit molecular descriptors as
features, with XGBoost and LightGBM as the primary models.

---

## 1. Dataset Overview

### 1.1 BBBP Dataset (Classification)

| Statistic | Value |
|-----------|-------|
| Source | MoleculeNet (Wu et al., 2018) |
| Total molecules | ~2050 |
| Valid (parsed) molecules | *see training log* |
| Invalid SMILES | *see data/invalid_smiles_report.txt* |
| Label: Permeable (1) | ~75% |
| Label: Non-permeable (0) | ~25% |
| Feature dimensions | 2057 (2048 fp bits + 9 descriptors) |
| Split | Random 70/15/15 (train/val/test) |

### 1.2 ESOL Dataset (Regression)

| Statistic | Value |
|-----------|-------|
| Source | Delaney (2004), curated by DeepChem |
| Total molecules | ~1128 |
| LogS range | −11.6 to +1.6 log(mol/L) |
| Mean LogS | −3.05 |
| Feature dimensions | 2057 (same as BBBP) |
| Split | Random 70/15/15 |

---

## 2. Feature Engineering Methodology

### 2.1 Morgan Fingerprints (ECFP4)

Morgan fingerprints encode the circular chemical neighbourhood of each atom:

```
AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
```

- **Radius 2** (ECFP4): captures atoms and their neighbours up to 2 bonds away
- **2048 bits**: bit-folded hash of substructure presence
- **Binary**: 0 = substructure absent, 1 = present
- Known limitation: hash collisions (different substructures → same bit)

### 2.2 Molecular Descriptors (RDKit)

| Descriptor | Chemical Relevance |
|------------|-------------------|
| MolWt | Size / membrane permeability (Lipinski ≤ 500 Da) |
| MolLogP | Lipophilicity — key for BBB penetration |
| TPSA | Polarity — TPSA < 90 Å² for CNS drugs |
| NumHDonors | Lipinski filter ≤ 5 |
| NumHAcceptors | Lipinski filter ≤ 10 |
| NumRotatableBonds | Molecular flexibility / oral absorption |
| RingCount | Structural rigidity |
| NumAromaticRings | π-system extent |
| FractionCSP3 | 3D character / aqueous solubility |

### 2.3 Invalid SMILES Handling

Molecules that failed `Chem.MolFromSmiles()` parsing were silently skipped.
Full list in `data/invalid_smiles_report.txt`.

---

## 3. Model Performance Comparison

### 3.1 Classification — BBBP (ROC-AUC, higher is better)

Random 70/15/15 split, stratified. 2039 valid molecules (11 invalid SMILES skipped).

| Model | CV AUC | Val AUC | Test AUC | Test F1 | Test Prec | Test Recall | Test Acc |
|-------|--------|---------|----------|---------|-----------|-------------|----------|
| Random Forest | 0.9123 | 0.9056 | **0.9539** | 0.9352 | 0.8885 | 0.9872 | 0.8954 |
| LightGBM | 0.9043 | 0.9069 | 0.9520 | 0.9356 | 0.9109 | 0.9615 | 0.8987 |
| XGBoost | 0.9092 | 0.9124 | 0.9490 | **0.9439** | **0.9190** | 0.9701 | **0.9118** |
| Logistic Regression | 0.9078 | 0.8839 | 0.9458 | 0.9333 | 0.8851 | 0.9872 | 0.8922 |

**Best ROC-AUC:** Random Forest (0.9539) | **Best F1/Accuracy:** XGBoost

See `outputs/model_comparison.csv` and `outputs/classification/` for plots.

### 3.2 Regression — ESOL (RMSE, lower is better)

1128 molecules, random 70/15/15 split. LogS range: -11.6 to +1.6 log(mol/L).

| Model | CV RMSE | Test RMSE | Test MAE | Test R² |
|-------|---------|-----------|----------|---------|
| LightGBM Regressor | 0.6543 | **0.6474** | 0.4859 | **0.8974** |
| XGBoost Regressor | 0.6804 | 0.6934 | **0.4803** | 0.8823 |
| Ridge Regression | 1.0966 | 0.9885 | 0.7660 | 0.7608 |

**Best:** LightGBM Regressor (RMSE=0.647, R²=0.897)

See `outputs/regression_results/` for plots.

---

## 4. SHAP Findings and Molecular Insights

### 4.1 Top Features for BBB Permeability (LightGBM SHAP TreeExplainer)

SHAP TreeExplainer computed on the full BBBP feature matrix (2039 molecules).

| Rank | Feature | Mean |SHAP| | Chemical Role |
|------|---------|------------|--------------|
| 1 | **TPSA** | 1.2145 | Polarity / desolvation cost |
| 2 | **MolLogP** | 0.6546 | Lipophilicity / membrane partitioning |
| 3 | **NumHDonors** | 0.5691 | H-bond capacity (reduces permeability) |
| 4 | FractionCSP3 | 0.4797 | 3D molecular shape / aqueous solubility |
| 5 | MolWt | 0.4554 | Size / desolvation penalty |

### 4.2 Key Insights

**TPSA** is the top predictor (mean |SHAP| = 1.21). This aligns strongly with
BBB pharmacokinetics: a TPSA > 90 Å² dramatically reduces CNS penetration
because polar atoms form hydrogen bonds with water, raising the energy cost
of partitioning from aqueous blood into the lipid bilayer of the BBB.

**MolLogP** (mean |SHAP| = 0.65) shows that moderate lipophilicity (1–3) is
optimal for CNS drugs. Too hydrophilic (LogP < 1) means the molecule prefers
aqueous phase; too lipophilic (LogP > 5) causes poor aqueous solubility and
increased binding to plasma proteins.

**NumHDonors** (mean |SHAP| = 0.57) reduces permeability because N-H and O-H
groups preferentially form hydrogen bonds with water, increasing the thermodynamic
cost of membrane partitioning (Lipinski: ≤ 5 donors for oral bioavailability;
CNS drugs typically need ≤ 3).

**FractionCSP3** (mean |SHAP| = 0.48) captures 3D molecular shape. Higher sp3
character is associated with better aqueous solubility and selectivity for CNS
targets, but reduces aromatic stacking which can be problematic for flat molecules.

---

## 5. Benchmark Comparison vs Literature

| Model | Features | Split | ROC-AUC |
|-------|----------|-------|---------|
| **Our: Logistic Regression** | ECFP4 + 9 desc | Random | **0.9458** |
| **Our: XGBoost** | ECFP4 + 9 desc | Random | **0.9490** |
| **Our: LightGBM** | ECFP4 + 9 desc | Random | **0.9520** |
| **Our: Random Forest** | ECFP4 + 9 desc | Random | **0.9539** |
| Literature: RF (MoleculeNet) | ECFP4 | Scaffold | 0.714 |
| Literature: GCN (MoleculeNet) | Graph | Scaffold | 0.877 |
| Literature: MPNN | Graph | Scaffold | 0.913 |
| Literature: AttentiveFP | Graph+Attn | Scaffold | 0.908 |

Our random-split results exceed the published scaffold-split baselines numerically.
This is primarily because random splits are easier (training and test molecules share
similar scaffolds). On a scaffold split, gradient-boosted ECFP4 models typically
achieve ~0.85–0.88, comparable to early GNN results.

See `outputs/benchmark_report.md` for full analysis.

---

## 6. Known Limitations and Failure Modes

### 6.1 Data Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| ~2050 molecules (BBBP) | Limited model capacity | Use ChEMBL / PubChem augmentation |
| Random split (not scaffold) | Overestimates real performance | Switch to scaffold split |
| Binary label (no assay data) | Noisy ground truth | Use quantitative assay data |
| Class imbalance (~75/25) | Recall bias for minority class | SMOTE / class weighting |

### 6.2 Feature Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| ECFP4 hash collisions | Feature aliasing | Larger n_bits (4096+) or atom-pair FP |
| No 3D features | Misses conformational effects | RDKit 3D descriptors, ROCS |
| Static SMILES features | No ionisation state handling | Protonate at pH 7.4 first |
| Radius=2 only captures local environment | Misses global topology | ECFP6 (radius=3) |

### 6.3 Model Failure Modes

- **Novel scaffolds:** Performance degrades on chemical series not represented
  in training data (generalisation gap).
- **Reactive molecules:** Molecules with unusual functional groups (e.g., acyl
  chlorides) may have valid SMILES but irrelevant descriptors.
- **Salt forms:** SMILES for salts (e.g., `[Na+].[Cl-]`) may need
  desalting/standardisation before inference.
- **Large molecules:** Peptides and macrocycles fall outside typical ECFP4
  coverage; dedicated models are needed.

---

## 7. Future Improvements

### 7.1 Algorithmic Improvements

1. **Graph Neural Networks** — Replace ECFP4 with GCN/MPNN operating directly
   on the molecular graph. Expected +5–15% AUC on scaffold split.

2. **Pre-trained Transformers** — Fine-tune ChemBERTa or MolBERT on BBBP.
   SMILES-based transformers capture long-range structure.

3. **3D Models** — Use Uni-Mol or EquiBind with 3D conformers for shape-based
   features (expect +10–20% on scaffold split).

4. **Multi-task Learning** — Co-train on related endpoints (LogP, solubility,
   CYP inhibition) to regularise representation learning.

### 7.2 Data Improvements

1. Augment with ChEMBL BBB data (~10k+ molecules)
2. Use scaffold-based splitting for honest evaluation
3. Include active permeability data (Pgp efflux ratio) as additional label
4. Standardise molecules (neutralise, remove salts) before featurisation

### 7.3 Production Improvements

1. Add SMILES standardisation (RDKit's `MolStandardize` or `chembl_structure_pipeline`)
2. Implement model uncertainty quantification (conformal prediction or MC dropout)
3. Add applicability domain detection (flag out-of-distribution queries)
4. Set up automated retraining with new experimental data
5. Add caching layer (Redis) for the FastAPI service

---

## 8. Key Plots Reference

| Plot | Location | Description |
|------|----------|-------------|
| Class balance | `outputs/eda/class_balance.png` | BBBP label distribution |
| Descriptor distributions | `outputs/eda/descriptor_distributions.png` | KDE plots by class |
| Correlation heatmap | `outputs/eda/descriptor_correlation_heatmap.png` | Descriptor correlations |
| Box plots by class | `outputs/eda/descriptor_boxplots_by_class.png` | Descriptor vs BBB class |
| Variable fingerprints | `outputs/eda/top_variable_fingerprints.png` | Top-20 variable bits |
| LogP vs MW scatter | `outputs/eda/logp_vs_molwt_scatter.png` | Chemical space overview |
| ROC curves (all) | `outputs/classification/roc_all_models.png` | All classifier ROC curves |
| SHAP summary | `outputs/shap/shap_summary_beeswarm.png` | Feature attribution beeswarm |
| SHAP bar | `outputs/shap/shap_bar_importance.png` | Mean |SHAP| importance |
| SHAP waterfall | `outputs/shap/shap_waterfall_single.png` | Single molecule explanation |
| Predicted vs actual | `outputs/regression_results/pred_vs_actual_*.png` | Solubility predictions |

---

## 9. References

1. Wu, Z. et al. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9(2), 513–530.
2. Delaney, J.S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. *J. Chem. Inf. Comput. Sci.*, 44(3), 1000–1005.
3. Rogers, D. & Hahn, M. (2010). Extended-connectivity fingerprints. *J. Chem. Inf. Model.*, 50(5), 742–754.
4. Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
5. Lipinski, C.A. et al. (1997). Experimental and computational approaches to estimate solubility and permeability in drug discovery. *Advanced Drug Delivery Reviews*, 23(1-3), 3–25.
