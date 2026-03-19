"""
Drug Discovery ML Pipeline — Main Runner
==========================================
Runs all pipeline steps in order:
  1. Feature engineering (download data + compute features)
  2. Exploratory data analysis
  3. Classification pipeline (BBBP)
  4. Regression pipeline (ESOL)
  5. SHAP interpretability
  6. Update final report with actual results

Usage
-----
    python run_pipeline.py [--data-dir data] [--output-dir outputs]

    # Skip specific steps (e.g. if features already computed):
    python run_pipeline.py --skip-features
    python run_pipeline.py --skip-eda
    python run_pipeline.py --skip-models
    python run_pipeline.py --only-api   # just print API start instructions
"""

import argparse
import os
import sys
import time
import traceback

# Ensure src/ is importable regardless of where this script is run from
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


# ── Progress Utilities ─────────────────────────────────────────────────────────

def banner(msg: str, char: str = "=") -> None:
    width = 62
    print("\n" + char * width)
    print(f"  {msg}")
    print(char * width)


def step(n: int, total: int, desc: str) -> float:
    print(f"\n[{n}/{total}] {desc}")
    return time.perf_counter()


def done(t0: float) -> None:
    elapsed = time.perf_counter() - t0
    print(f"  [DONE] in {elapsed:.1f}s")


def fail(step_name: str, exc: Exception) -> None:
    print(f"\n  [ERROR] Step '{step_name}' failed:")
    print(f"    {type(exc).__name__}: {exc}")
    print("  Traceback:")
    traceback.print_exc()
    print("\n  Pipeline continuing with remaining steps...")


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Drug Discovery Molecular Property Prediction Pipeline"
    )
    parser.add_argument("--data-dir",    default="data",    help="Data directory")
    parser.add_argument("--output-dir",  default="outputs", help="Output directory")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature engineering (use existing CSVs)")
    parser.add_argument("--skip-eda",    action="store_true",
                        help="Skip EDA step")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip model training (use existing .pkl files)")
    parser.add_argument("--skip-shap",   action="store_true",
                        help="Skip SHAP analysis")
    parser.add_argument("--only-api",    action="store_true",
                        help="Print API startup instructions and exit")
    args = parser.parse_args()

    if args.only_api:
        print("\nTo start the FastAPI service, run:\n")
        print("  uvicorn src.api:app --reload --port 8000\n")
        print("Then visit http://localhost:8000/docs")
        return

    banner("Drug Discovery Molecular Property Prediction Pipeline", "=")
    print(f"  Data directory  : {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")

    total_steps = 5
    pipeline_t0 = time.perf_counter()

    # ── Step 1: Feature Engineering ───────────────────────────────────────────
    if not args.skip_features:
        t0 = step(1, total_steps, "Feature Engineering (download + ECFP4 + descriptors)")
        try:
            from src.feature_engineering import run_feature_engineering
            run_feature_engineering(data_dir=args.data_dir)
            done(t0)
        except Exception as e:
            fail("Feature Engineering", e)
    else:
        print(f"\n[1/{total_steps}] Feature Engineering — SKIPPED (--skip-features)")

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    if not args.skip_eda:
        t0 = step(2, total_steps, "Exploratory Data Analysis")
        bbbp_features = os.path.join(args.data_dir, "features_bbbp.csv")
        if not os.path.exists(bbbp_features):
            print(f"  [WARN] {bbbp_features} not found. Skipping EDA.")
        else:
            try:
                from src.eda import run_eda
                run_eda(
                    features_path=bbbp_features,
                    invalid_report_path=os.path.join(args.data_dir,
                                                      "invalid_smiles_report.txt"),
                    out_dir=os.path.join(args.output_dir, "eda"),
                    summary_path=os.path.join(args.output_dir, "eda_summary.md"),
                )
                done(t0)
            except Exception as e:
                fail("EDA", e)
    else:
        print(f"\n[2/{total_steps}] EDA — SKIPPED (--skip-eda)")

    # ── Step 3+4: Model Training ───────────────────────────────────────────────
    if not args.skip_models:
        t0 = step(3, total_steps, "Classification Pipeline (BBBP)")
        bbbp_features = os.path.join(args.data_dir, "features_bbbp.csv")
        if not os.path.exists(bbbp_features):
            print(f"  [WARN] {bbbp_features} not found. Skipping classification.")
        else:
            try:
                from src.models import run_classification_pipeline
                run_classification_pipeline(
                    features_path=bbbp_features,
                    out_dir=args.output_dir,
                    models_dir=os.path.join(args.output_dir, "models"),
                )
                done(t0)
            except Exception as e:
                fail("Classification", e)

        t0 = step(4, total_steps, "Regression Pipeline (ESOL)")
        esol_features = os.path.join(args.data_dir, "features_esol.csv")
        if not os.path.exists(esol_features):
            print(f"  [WARN] {esol_features} not found. Skipping regression.")
        else:
            try:
                from src.models import run_regression_pipeline
                run_regression_pipeline(
                    features_path=esol_features,
                    out_dir=os.path.join(args.output_dir, "regression_results"),
                    models_dir=os.path.join(args.output_dir, "models"),
                )
                done(t0)
            except Exception as e:
                fail("Regression", e)
    else:
        print(f"\n[3/{total_steps}] Classification — SKIPPED (--skip-models)")
        print(f"[4/{total_steps}] Regression    — SKIPPED (--skip-models)")

    # ── Step 5: SHAP ──────────────────────────────────────────────────────────
    if not args.skip_shap:
        t0 = step(5, total_steps, "SHAP Interpretability")

        # Find XGBoost model
        models_dir = os.path.join(args.output_dir, "models")
        ref_file   = os.path.join(models_dir, "best_xgb_path.txt")
        xgb_path   = os.path.join(models_dir, "xgboost.pkl")

        if os.path.exists(ref_file):
            with open(ref_file) as f:
                candidate = f.read().strip()
            if os.path.exists(candidate):
                xgb_path = candidate

        if not os.path.exists(xgb_path):
            print(f"  [WARN] XGBoost model not found at {xgb_path}.")
            print("  Skipping SHAP. Run model training first.")
        else:
            try:
                from src.shap_analysis import run_shap_analysis
                run_shap_analysis(
                    model_path=xgb_path,
                    features_path=os.path.join(args.data_dir, "features_bbbp.csv"),
                    out_dir=os.path.join(args.output_dir, "shap"),
                )
                done(t0)
            except Exception as e:
                fail("SHAP", e)
    else:
        print(f"\n[5/{total_steps}] SHAP — SKIPPED (--skip-shap)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - pipeline_t0
    banner(f"Pipeline Complete  ({total_elapsed:.0f}s total)")

    comp_path = os.path.join(args.output_dir, "model_comparison.csv")
    if os.path.exists(comp_path):
        import pandas as pd
        comp = pd.read_csv(comp_path)
        print("\n-- Classification Results (BBBP) --")
        print(comp[["model", "test_auc", "test_f1", "test_acc"]].to_string(index=False))

    reg_path = os.path.join(args.output_dir, "regression_results",
                             "regression_comparison.csv")
    if os.path.exists(reg_path):
        import pandas as pd
        reg = pd.read_csv(reg_path)
        print("\n-- Regression Results (ESOL) --")
        print(reg[["model", "test_rmse", "test_mae", "test_r2"]].to_string(index=False))

    print("\n-- Key output files --")
    artifacts = [
        ("Feature matrix (BBBP)", f"{args.data_dir}/features_bbbp.csv"),
        ("Feature matrix (ESOL)", f"{args.data_dir}/features_esol.csv"),
        ("Data dictionary",       f"{args.data_dir}/data_dictionary.md"),
        ("EDA summary",           f"{args.output_dir}/eda_summary.md"),
        ("Model comparison",      f"{args.output_dir}/model_comparison.csv"),
        ("SHAP interpretation",   f"{args.output_dir}/shap/shap_interpretation.md"),
        ("Benchmark report",      f"{args.output_dir}/benchmark_report.md"),
        ("Final report",          f"{args.output_dir}/final_report.md"),
    ]
    for label, path in artifacts:
        exists = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"  {exists:9s}  {label:30s}  {path}")

    print("\n-- MLflow UI --")
    print("  mlflow ui --port 5000  ->  http://localhost:5000")

    print("\n-- FastAPI Service --")
    print("  uvicorn src.api:app --reload --port 8000")
    print("  -> http://localhost:8000/docs\n")


if __name__ == "__main__":
    main()
