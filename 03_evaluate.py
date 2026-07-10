"""
03_evaluate.py  —  Results Evaluation & Figure Generation
Paper: Towards Reproducible AI-Generated Scientific Visualizations
Author: Diana Ticudean, Technical University of Cluj-Napoca, 2026

Loads the trained mapping model and results CSV, then produces:
  1. Figure 1 — Fidelity score distribution per output type (histogram)
  2. Figure 2 — FPAR comparison: baseline vs model-guided (bar chart)
  3. Figure 3 — Feature importance plot (top 20 seed bits)
  4. Figure 4 — Acceptance rate heatmap: seed (x) × prompt (y)
  5. paper_values.json — all [VALUE] placeholders filled in for Section 4

Run after 02_train_mapping_model.py.
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR          = Path(__file__).parent
RESULTS_CSV       = BASE_DIR / "experiment" / "results.csv"
MODEL_DIR         = BASE_DIR / "experiment" / "model"
FIGURES_DIR       = BASE_DIR / "experiment" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ACCEPTANCE_THRESHOLD = 0.70

OUTPUT_TYPES = [
    "bar_chart", "line_chart", "scatter_plot",
    "network_diagram", "flowchart"
]
TYPE_LABELS = {
    "bar_chart"      : "Bar Chart",
    "line_chart"     : "Line Chart",
    "scatter_plot"   : "Scatter Plot",
    "network_diagram": "Network Diagram",
    "flowchart"      : "Flowchart",
}
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def seed_to_bits(seed):
    return [(seed >> i) & 1 for i in range(32)]

def build_features(df, le):
    seed_bits   = np.array([seed_to_bits(s) for s in df["seed"]], dtype=np.float32)
    type_enc    = le.transform(df["output_type"])
    type_onehot = np.zeros((len(df), len(OUTPUT_TYPES)), dtype=np.float32)
    for i, t in enumerate(type_enc):
        type_onehot[i, t] = 1.0
    return np.hstack([seed_bits, type_onehot])

def feature_names():
    return [f"seed_bit_{i:02d}" for i in range(32)] + [f"type_{t}" for t in OUTPUT_TYPES]

# ── Figure 1: Fidelity Score Distribution ─────────────────────────────────────

def fig_fidelity_distribution(df):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=False)
    fig.suptitle("Fidelity Score Distribution by Output Type", fontsize=13, fontweight="bold")

    for ax, ot, color in zip(axes, OUTPUT_TYPES, COLORS):
        scores = df[df["output_type"] == ot]["fidelity_score"]
        ax.hist(scores, bins=20, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(ACCEPTANCE_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
                   label=f"Threshold ({ACCEPTANCE_THRESHOLD})")
        ax.set_title(TYPE_LABELS[ot], fontsize=10, fontweight="bold")
        ax.set_xlabel("Fidelity Score", fontsize=9)
        ax.set_ylabel("Count" if ot == OUTPUT_TYPES[0] else "", fontsize=9)
        acc_rate = (scores >= ACCEPTANCE_THRESHOLD).mean()
        ax.text(0.97, 0.97, f"FPAR={acc_rate:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "fig1_fidelity_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")
    return path

# ── Figure 2: FPAR Comparison ─────────────────────────────────────────────────

def fig_fpar_comparison(df, model, le):
    X_all = build_features(df, le)
    df = df.copy()
    df["predicted"] = model.predict(X_all)

    baseline_fpars = []
    model_fpars    = []
    labels         = []

    for ot in OUTPUT_TYPES:
        subset     = df[df["output_type"] == ot]
        predicted  = subset[subset["predicted"] == 1]
        baseline_fpars.append(subset["above_threshold"].mean())
        model_fpars.append(predicted["above_threshold"].mean() if len(predicted) else 0.0)
        labels.append(TYPE_LABELS[ot])

    x     = np.arange(len(OUTPUT_TYPES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, baseline_fpars, width, label="Baseline (random seed)",
                   color="#90A4AE", edgecolor="white")
    bars2 = ax.bar(x + width/2, model_fpars,    width, label="Model-guided selection",
                   color="#1565C0", edgecolor="white")

    ax.set_ylabel("First-Pass Acceptance Rate (FPAR)", fontsize=11)
    ax.set_title("FPAR: Baseline vs. Seed-Structure Mapping Model", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color="#1565C0")

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_fpar_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")
    return path, baseline_fpars, model_fpars

# ── Figure 3: Feature Importance ──────────────────────────────────────────────

def fig_feature_importance(model):
    importances = model.feature_importances_
    names       = feature_names()
    feat_df     = pd.DataFrame({"feature": names, "importance": importances})
    feat_df     = feat_df.sort_values("importance", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors  = ["#1565C0" if "bit" in n else "#F44336" for n in feat_df["feature"]]
    ax.barh(feat_df["feature"], feat_df["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (XGBoost gain)", fontsize=11)
    ax.set_title("Top 20 Feature Importances", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#1565C0", label="Seed bit"),
              Patch(facecolor="#F44336",  label="Output type")]
    ax.legend(handles=legend, fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")
    return path

# ── Figure 4: Acceptance Heatmap ──────────────────────────────────────────────

def fig_acceptance_heatmap(df):
    """Per-prompt acceptance rate heatmap (output_type × prompt_idx)."""
    pivot = df.groupby(["output_type", "prompt_idx"])["above_threshold"].mean().unstack()
    pivot = pivot.reindex(OUTPUT_TYPES)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Acceptance Rate")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"P{i}" for i in range(pivot.shape[1])], fontsize=9)
    ax.set_yticks(range(len(OUTPUT_TYPES)))
    ax.set_yticklabels([TYPE_LABELS[t] for t in OUTPUT_TYPES], fontsize=9)
    ax.set_xlabel("Prompt Index", fontsize=11)
    ax.set_title("Acceptance Rate Heatmap (Output Type × Prompt)", fontsize=12, fontweight="bold")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.tight_layout()
    path = FIGURES_DIR / "fig4_acceptance_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")
    return path

# ── Paper Values ──────────────────────────────────────────────────────────────

def extract_paper_values(df, model, le, baseline_fpars, model_fpars, cv_scores=None):
    """
    Extract all [VALUE] placeholders needed for Section 4 of the paper.
    Saves to experiment/model/paper_values.json
    """
    X_all = build_features(df, le)
    df = df.copy()
    df["predicted"] = model.predict(X_all)

    overall_baseline = df["above_threshold"].mean()
    predicted_pos    = df[df["predicted"] == 1]
    overall_model    = predicted_pos["above_threshold"].mean() if len(predicted_pos) else 0.0
    improvement_pp   = (overall_model - overall_baseline) * 100

    feat_imp         = pd.DataFrame({
        "feature": feature_names(),
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    top_bit = feat_imp[feat_imp["feature"].str.startswith("seed_bit")].iloc[0]["feature"]
    top_bit_importance = feat_imp.iloc[0]["importance"]

    # Best and worst output types
    type_fpars = {ot: df[df["output_type"] == ot]["above_threshold"].mean() for ot in OUTPUT_TYPES}
    best_type  = max(type_fpars, key=type_fpars.get)
    worst_type = min(type_fpars, key=type_fpars.get)

    values = {
        "total_images"              : len(df),
        "seeds_per_prompt"          : int(df["seed"].nunique()),
        "output_types_count"        : int(df["output_type"].nunique()),
        "prompts_per_type"          : int(df["prompt_idx"].nunique()),
        "overall_baseline_fpar"     : round(float(overall_baseline), 4),
        "overall_baseline_fpar_pct" : round(float(overall_baseline) * 100, 1),
        "overall_model_fpar"        : round(float(overall_model), 4),
        "overall_model_fpar_pct"    : round(float(overall_model) * 100, 1),
        "improvement_pp"            : round(float(improvement_pp), 1),
        "seeds_flagged_good"        : int(len(predicted_pos)),
        "seeds_flagged_good_pct"    : round(100 * len(predicted_pos) / len(df), 1),
        "best_output_type"          : TYPE_LABELS[best_type],
        "best_type_fpar"            : round(float(type_fpars[best_type]), 4),
        "worst_output_type"         : TYPE_LABELS[worst_type],
        "worst_type_fpar"           : round(float(type_fpars[worst_type]), 4),
        "top_feature"               : top_bit,
        "top_feature_importance"    : round(float(top_bit_importance), 4),
        "per_type_baseline_fpar"    : {ot: round(float(v), 4) for ot, v in type_fpars.items()},
        "per_type_model_fpar"       : {ot: round(float(v), 4) for ot, v in
                                        zip(OUTPUT_TYPES, model_fpars)},
    }

    if cv_scores is not None:
        values["cv_f1_mean"] = round(float(np.mean(cv_scores)), 4)
        values["cv_f1_std"]  = round(float(np.std(cv_scores)), 4)

    out_path = MODEL_DIR / "paper_values.json"
    with open(out_path, "w") as f:
        json.dump(values, f, indent=2)

    print(f"\n  Saved: {out_path.name}")
    print(f"\nKey values for Section 4:")
    print(f"  Overall baseline FPAR : {values['overall_baseline_fpar_pct']}%")
    print(f"  Model-guided FPAR     : {values['overall_model_fpar_pct']}%")
    print(f"  Improvement           : +{values['improvement_pp']} pp")
    print(f"  Best output type      : {values['best_output_type']} ({values['best_type_fpar']:.4f})")
    print(f"  Worst output type     : {values['worst_output_type']} ({values['worst_type_fpar']:.4f})")

    return values

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import xgboost as xgb

    print("=" * 60)
    print("Seed-Structure Mapping — Evaluation & Figure Generation")
    print("=" * 60)

    # Load data
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found.")
        return
    df = pd.read_csv(RESULTS_CSV)
    print(f"\nLoaded {len(df):,} results")

    # Load model
    model_path = MODEL_DIR / "mapping_model.json"
    le_path    = MODEL_DIR / "label_encoder.pkl"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run 02_train_mapping_model.py first.")
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    print("Model loaded.")

    # Generate figures
    print("\nGenerating figures ...")
    fig_fidelity_distribution(df)
    _, baseline_fpars, model_fpars = fig_fpar_comparison(df, model, le)
    fig_feature_importance(model)
    fig_acceptance_heatmap(df)

    # Extract paper values
    print("\nExtracting paper values ...")
    extract_paper_values(df, model, le, baseline_fpars, model_fpars)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Open experiment/model/paper_values.json to fill Section 4 [VALUE] placeholders.")
    print("\nDone.")

if __name__ == "__main__":
    main()
