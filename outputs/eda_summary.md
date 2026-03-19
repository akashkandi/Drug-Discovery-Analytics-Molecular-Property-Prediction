# BBBP Dataset — EDA Summary

**Generated on:** 2026-03-18


## 1. Dataset Overview
| Item | Value |
|------|-------|
| Total molecules in CSV | 2050 |
| Valid molecules (parsed) | 2039 |
| Invalid SMILES (skipped) | 11 |
| Feature dimensions | 2060 columns |
| Morgan fingerprint bits | 2048 (ECFP4, radius=2) |
| Molecular descriptors | 9 |

## 2. Class Balance
| Class | Count | Percentage |
|-------|-------|------------|
| Non-permeable (0) | 479 | 23.5% |
| Permeable (1) | 1560 | 76.5% |

## 3. Molecular Descriptor Statistics

|       |    MolWt |   MolLogP |     TPSA |   NumHDonors |   NumHAcceptors |   NumRotatableBonds |   RingCount |   NumAromaticRings |   FractionCSP3 |
|:------|---------:|----------:|---------:|-------------:|----------------:|--------------------:|------------:|-------------------:|---------------:|
| count | 2039     |  2039     | 2039     |     2039     |        2039     |            2039     |    2039     |           2039     |       2039     |
| mean  |  344.536 |     2.318 |   70.728 |        1.546 |           4.443 |               4.199 |       2.991 |              1.413 |          0.462 |
| std   |  150.654 |     2.093 |   58.469 |        1.841 |           3.2   |               3.044 |       1.55  |              0.997 |          0.231 |
| min   |   28.054 |   -11.745 |    0     |        0     |           0     |               0     |       0     |              0     |          0     |
| 25%   |  256.81  |     1.254 |   32.78  |        0     |           2     |               2     |       2     |              1     |          0.3   |
| 50%   |  324.406 |     2.475 |   55.12  |        1     |           4     |               4     |       3     |              1     |          0.429 |
| 75%   |  410.767 |     3.746 |   92.87  |        2     |           5     |               6     |       4     |              2     |          0.611 |
| max   | 1879.68  |    10.813 |  662.41  |       24     |          33     |              35     |      16     |              7     |          1     |

## 4. Key Observations

- **Class imbalance:** The BBBP dataset is moderately imbalanced, with more permeable molecules (~75%) than non-permeable ones.
- **LogP range:** Most molecules have LogP between -2 and 8, consistent with CNS-active drug-like space.
- **Molecular weight:** Majority of molecules fall below 500 Da (Lipinski Rule of 5 boundary).
- **TPSA:** CNS-active molecules typically have TPSA < 90 Å², which is reflected in the distribution.
- **Hydrogen bond donors/acceptors:** Both distributions are right-skewed, consistent with drug-like molecules.

## 5. Generated Plots

| Plot | File |
|------|------|
| Class balance bar chart | `outputs/eda/class_balance.png` |
| Descriptor distributions by class | `outputs/eda/descriptor_distributions.png` |
| Descriptor correlation heatmap | `outputs/eda/descriptor_correlation_heatmap.png` |
| Box plots by class | `outputs/eda/descriptor_boxplots_by_class.png` |
| Top-20 variable fingerprint bits | `outputs/eda/top_variable_fingerprints.png` |
| LogP vs MolWt scatter | `outputs/eda/logp_vs_molwt_scatter.png` |

## 6. Invalid SMILES Report
- 11 molecules could not be parsed by RDKit.
- These were silently excluded from the feature matrix.
- Full list saved to `data/invalid_smiles_report.txt`.