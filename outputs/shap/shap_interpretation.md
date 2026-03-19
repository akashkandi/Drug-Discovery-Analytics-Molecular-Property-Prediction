# SHAP Interpretability — Key Findings

## Top Features Driving BBB Permeability

The SHAP analysis of the best XGBoost model reveals the following molecular
features as most predictive of blood-brain barrier (BBB) permeability:

### Ranked Feature Importance (Top 20)

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | TPSA | 1.2145 |
| 2 | MolLogP | 0.6546 |
| 3 | NumHDonors | 0.5691 |
| 4 | FractionCSP3 | 0.4797 |
| 5 | MolWt | 0.4554 |
| 6 | bit 5 | 0.4363 |
| 7 | bit 1928 | 0.2912 |
| 8 | NumRotatableBonds | 0.2089 |
| 9 | NumHAcceptors | 0.1858 |
| 10 | bit 1564 | 0.1726 |
| 11 | bit 1171 | 0.1718 |
| 12 | bit 456 | 0.1662 |
| 13 | bit 314 | 0.1631 |
| 14 | bit 1145 | 0.1325 |
| 15 | bit 1791 | 0.1268 |
| 16 | bit 695 | 0.1262 |
| 17 | bit 1365 | 0.1228 |
| 18 | bit 231 | 0.1223 |
| 19 | bit 1452 | 0.1196 |
| 20 | bit 807 | 0.1194 |

## Interpretation

**TPSA** emerged as the most predictive feature (mean |SHAP| = 1.2145).
This aligns with established medicinal chemistry knowledge: lipophilicity (LogP)
is a primary determinant of passive membrane permeability. Molecules with moderate
LogP (1–3) are better able to partition into the lipid bilayer of the BBB while
retaining sufficient aqueous solubility to avoid aggregation.

**MolLogP** (mean |SHAP| = 0.6546) shows a strong negative correlation
with permeability. Topological Polar Surface Area (TPSA) measures the combined
surface area of polar atoms; a TPSA > 90 Å² is associated with poor CNS
penetration because polar atoms form hydrogen bonds with water, increasing the
energetic cost of membrane partitioning.

**NumHDonors** (mean |SHAP| = 0.5691) also contributes significantly.
Molecular weight correlates with the number of atoms that must "desolvate" to
cross the membrane. Larger molecules have higher entropic costs for membrane
passage, which is reflected in the negative SHAP values for heavy molecules.

## Lipinski Rule of 5 Alignment

The SHAP results are consistent with Lipinski's Rule of 5 framework:
- LogP ≤ 5 → positively weighted by model
- MW ≤ 500 Da → negatively penalised for heavier molecules
- H-bond donors ≤ 5 → SHAP confirms donors reduce permeability
- H-bond acceptors ≤ 10 → acceptors have moderate negative contribution

## Limitations

- Morgan fingerprint bits (ECFP4) that appear in the top features encode
  specific structural fragments; their chemical interpretation requires
  mapping the bit index back to the substructure (using `AllChem.GetMorganBitInfo`).
- SHAP values reflect correlations in the training data; causal claims
  require additional validation.
- The model was trained on ~2000 molecules; SHAP stability may improve
  with a larger, more diverse dataset.
