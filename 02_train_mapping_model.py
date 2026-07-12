"""
02_train_mapping_model.py  —  Seed-Structure Mapping Model Training
Paper: Towards Reproducible AI-Generated Scientific Visualizations
Author: Diana Ticudean, Technical University of Cluj-Napoca, 2026

Trains an XGBoost classifier to predict whether a given (seed, output_type)
combination will produce a structurally faithful image (fidelity >= 0.70),
without generating the image first.

Input : experiment/results.csv  (produced by 01_generate_and_score.py)
Output: experiment/model/
          mapping_model.json        — trained XGBoost model
          label_encoder.pkl         — encodes output_type strings to ints
          feature_importance.csv    — ranked feature importances
          training_report.txt       — accuracy, precision, recall, F1, FPAR

Features (37 total):
  - 32 binary features: bits 0–31 of the seed integer
  - 5  binary features: one-hot encoding of output_type
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import time
import xgboost as xgb

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR          = Path(__file__).parent
RESULTS_CSV       = BASE_DIR / "experiment" / "results.csv"
MODEL_DIR         = BASE_DIR / "experiment" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ACCEPTANCE_THRESHOLD = 0.70   # must match 01_generate_and_score.py
TEST_SIZE            = 0.20   # 80/20 train/test split
RANDOM_STATE         = 42
CV_FOLDS             = 5

OUTPUT_TYPES = [
    "bar_chart", "line_chart", "scatter_plot",
    "network_diagram", "flowchart"
]

# ── Feature Engineering ───────────────────────────────────────────────────────

def seed_to_bits(seed: int) -> list:
    """Decompose a 32-bit seed integer into 32 binary features."""
    return [(seed >> i) & 1 for i in range(32)]

def build_features(df: pd.DataFrame, le: LabelEncoder) -> np.ndarray:
    """
    Build feature matrix X from results DataFrame.
    Returns array of shape (n_samples, 37):
      cols 0–31 : seed bits
      cols 32–36: one-hot output_type
    """
    seed_bits  = np.array([seed_to_bits(s) for s in df["seed"]], dtype=np.float32)
    type_enc   = le.transform(df["output_type"])
    type_onehot = np.zeros((len(df), len(OUTPUT_TYPES)), dtype=np.float32)
    for i, t in enumerate(type_enc):
        type_onehot[i, t] = 1.0
    return np.hstack([seed_bits, type_onehot])

def feature_names() -> list:
    bit_names  = [f"seed_bit_{i:02d}" for i in range(32)]
    type_names = [f"type_{t}" for t in OUTPUT_TYPES]
    return bit_names + type_names

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Seed-Structure Mapping Model — Training")
    print("=" * 60)

    # 1. Load data
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run 01_generate_and_score.py first.")
        return

    df = pd.read_csv(RESULTS_CSV)
    print(f"\nDataset loaded: {len(df):,} samples")
    print(f"  Output types : {df['output_type'].nunique()}")
    print(f"  Seed range   : {df['seed'].min()} – {df['seed'].max()}")
    print(f"  Accepted (≥{ACCEPTANCE_THRESHOLD}): {df['above_threshold'].sum():,} "
          f"({100*df['above_threshold'].mean():.1f}%)")

    # Per-type acceptance rates
    print("\nAcceptance rate per output type:")
    for ot in OUTPUT_TYPES:
        subset = df[df["output_type"] == ot]
        if len(subset):
            rate = subset["above_threshold"].mean()
            print(f"  {ot:<20} {rate:.3f}  ({subset['above_threshold'].sum()}/{len(subset)})")

    # 2. Build features
    le = LabelEncoder()
    le.fit(OUTPUT_TYPES)

    X = build_features(df, le)
    y = df["above_threshold"].values.astype(int)
    print(f"\nFeature matrix: {X.shape}  (37 features per sample)")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,} samples  |  Test: {len(X_test):,} samples")

    # 4. Train XGBoost
    print("\nTraining XGBoost classifier ...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators       = 500,
        max_depth          = 6,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = scale_pos_weight,
        use_label_encoder  = False,
        eval_metric        = "logloss",
        random_state       = RANDOM_STATE,
        n_jobs             = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print("Training complete.")

    # 5. Cross-validation
    print(f"\nCross-validation ({CV_FOLDS}-fold, stratified) ...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 6. Test set evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nTest set results:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"\nClassification report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    # 6b. AUC-ROC for XGBoost
    y_prob_xgb = model.predict_proba(X_test)[:, 1]
    auc_xgb    = roc_auc_score(y_test, y_prob_xgb)
    print(f"  AUC-ROC (XGBoost) : {auc_xgb:.4f}")

    # 6c. Train MLP and Logistic Regression for comparison
    print("\nTraining comparison models ...")

    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128, 128), activation="relu",
                        max_iter=300, random_state=RANDOM_STATE)
    mlp.fit(X_train, y_train)
    latency_mlp = (time.time() - t0) * 1000 / len(X_test)

    t0 = time.time()
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(X_train, y_train)
    latency_lr = (time.time() - t0) * 1000 / len(X_test)
    latency_xgb_ms = None  # measured below

    def eval_model(m, name, X_te, y_te):
        yp   = m.predict(X_te)
        yprob = m.predict_proba(X_te)[:, 1]
        t0 = time.time()
        for _ in range(100): m.predict(X_te[:1000])
        lat = (time.time() - t0) / 100 * 1000  # ms per 1000 samples
        return {
            "name"     : name,
            "accuracy" : round(float(accuracy_score(y_te, yp)), 4),
            "precision": round(float(precision_score(y_te, yp, zero_division=0)), 4),
            "recall"   : round(float(recall_score(y_te, yp, zero_division=0)), 4),
            "f1"       : round(float(f1_score(y_te, yp, zero_division=0)), 4),
            "auc_roc"  : round(float(roc_auc_score(y_te, yprob)), 4),
            "latency_ms_1000": round(lat, 2),
        }

    res_xgb = eval_model(model, "XGBoost",            X_test, y_test)
    res_mlp = eval_model(mlp,   "MLP (128-128)",       X_test, y_test)
    res_lr  = eval_model(lr,    "Logistic Regression", X_test, y_test)

    comparison = [res_xgb, res_mlp, res_lr]
    print("\nModel comparison (test set):")
    print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'ms/1k':>8}")
    for r in comparison:
        print(f"  {r['name']:<22} {r['accuracy']:>6.4f} {r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} {r['f1']:>6.4f} {r['auc_roc']:>6.4f} "
              f"{r['latency_ms_1000']:>8.2f}")

    comp_path = MODEL_DIR / "model_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved: {comp_path.name}")

    # 7. FPAR simulation
    # Simulate: instead of random seed, use model to pre-select seeds
    # For each (output_type, prompt_id), what % of model-selected seeds are accepted?
    print("\nFirst-Pass Acceptance Rate (FPAR) simulation:")
    baseline_fpar = df["above_threshold"].mean()

    # Build all features for full dataset
    X_all = build_features(df, le)
    df["predicted_accept"] = model.predict(X_all)

    # FPAR when only using model-predicted-positive seeds
    predicted_positive = df[df["predicted_accept"] == 1]
    if len(predicted_positive):
        model_fpar = predicted_positive["above_threshold"].mean()
    else:
        model_fpar = 0.0

    print(f"  Baseline FPAR (random seed)     : {baseline_fpar:.4f} ({baseline_fpar*100:.1f}%)")
    print(f"  Model-guided FPAR               : {model_fpar:.4f} ({model_fpar*100:.1f}%)")
    print(f"  Improvement                     : +{(model_fpar - baseline_fpar)*100:.1f} pp")
    print(f"  Seeds flagged as good           : {len(predicted_positive):,} / {len(df):,} "
          f"({100*len(predicted_positive)/len(df):.1f}%)")

    # 8. Feature importance
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "feature"   : feature_names(),
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(f"\nTop 10 most important features:")
    print(feat_df.head(10).to_string(index=False))

    # 9. Save outputs
    model.save_model(MODEL_DIR / "mapping_model.json")

    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    feat_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    # Training report
    report_lines = [
        "Seed-Structure Mapping Model — Training Report",
        "=" * 50,
        f"Dataset          : {len(df):,} samples",
        f"Output types     : {df['output_type'].nunique()}",
        f"Seed range       : {df['seed'].min()} – {df['seed'].max()}",
        f"Acceptance thresh: {ACCEPTANCE_THRESHOLD}",
        f"Baseline FPAR    : {baseline_fpar:.4f} ({baseline_fpar*100:.1f}%)",
        "",
        "Model: XGBoost (n_estimators=500, max_depth=6, lr=0.05)",
        f"CV F1 ({CV_FOLDS}-fold)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        "",
        "Test Set Results:",
        f"  Accuracy       : {acc:.4f}",
        f"  Precision      : {prec:.4f}",
        f"  Recall         : {rec:.4f}",
        f"  F1             : {f1:.4f}",
        "",
        "FPAR Simulation:",
        f"  Baseline FPAR  : {baseline_fpar:.4f}",
        f"  Model FPAR     : {model_fpar:.4f}",
        f"  Improvement    : +{(model_fpar - baseline_fpar)*100:.1f} pp",
        f"  Seeds selected : {len(predicted_positive):,} / {len(df):,}",
        "",
        "Top 10 Features:",
        feat_df.head(10).to_string(index=False),
    ]

    report_path = MODEL_DIR / "training_report.txt"
    report_path.write_text("\n".join(report_lines))

    print(f"\nOutputs saved to {MODEL_DIR}/")
    print(f"  mapping_model.json")
    print(f"  label_encoder.pkl")
    print(f"  feature_importance.csv")
    print(f"  training_report.txt")
    print("\nNext: run 03_evaluate.py to generate paper figures.")

if __name__ == "__main__":
    main()
