
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# Zerve Design System
_BG = "#1D1D20"
_TXT = "#fbfbff"
_TXT2 = "#909094"
_COLS = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
         "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
_HL = "#ffd400"
_SUCCESS = "#17b26a"
_WARN = "#f04438"

_TARGETS = ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]
_TARGET_LABELS = {
    "y_ret_30d": "30-Day Retention",
    "y_ret_90d": "90-Day Retention",
    "y_upgrade_60d": "60-Day Upgrade"
}
_MODEL_KEYS = ["LR_single", "L2_LR", "GBT"]
_MODEL_NAMES = {
    "LR_single": "LR (active_days)",
    "L2_LR": "L2 Logistic Reg.",
    "GBT": "Gradient-Boosted Trees"
}
_MODEL_COLORS = {
    "LR_single": _COLS[0],
    "L2_LR": _COLS[1],
    "GBT": _COLS[2]
}

_test_mask = modeling_df["split"] == "test"

# ═════════════════════════════════════════════════════
# 1. CALIBRATION PLOTS — one per target
# ═════════════════════════════════════════════════════
fig_calibration, _cal_axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=_BG)

for _idx, _target in enumerate(_TARGETS):
    _ax = _cal_axes[_idx]
    _ax.set_facecolor(_BG)
    
    _y_test = modeling_df.loc[_test_mask, _target].values
    
    # Perfect calibration line
    _ax.plot([0, 1], [0, 1], linestyle="--", color=_TXT2, linewidth=1, alpha=0.7, label="Perfect")
    
    for _mk in _MODEL_KEYS:
        _probs = all_predictions[(_target, _mk)]
        _n_bins = 8 if _target != "y_upgrade_60d" else 5  # fewer bins for rare event
        
        _frac_pos, _mean_pred = calibration_curve(_y_test, _probs, n_bins=_n_bins, strategy="quantile")
        
        _ax.plot(_mean_pred, _frac_pos, marker="o", markersize=5, linewidth=2,
                 color=_MODEL_COLORS[_mk], label=_MODEL_NAMES[_mk])
    
    _ax.set_title(_TARGET_LABELS[_target], color=_TXT, fontsize=13, fontweight="bold", pad=10)
    _ax.set_xlabel("Predicted Probability", color=_TXT, fontsize=11)
    if _idx == 0:
        _ax.set_ylabel("Observed Fraction Positive", color=_TXT, fontsize=11)
    _ax.tick_params(colors=_TXT2, labelsize=9)
    _ax.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=8, loc="upper left")
    _ax.set_xlim(-0.02, 1.02)
    _ax.set_ylim(-0.02, 1.02)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["bottom"].set_color(_TXT2)
    _ax.spines["left"].set_color(_TXT2)

plt.suptitle("Model Calibration Plots (Test Set)", color=_TXT, fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# ═════════════════════════════════════════════════════
# 2. MODEL COMPARISON CHART — grouped bar chart of all metrics
# ═════════════════════════════════════════════════════
_METRICS = ["ROC-AUC", "PR-AUC", "Lift@10%", "Lift@20%"]
_ALL_MODELS = ["Constant (base rate)", "LR (active_days only)", "L2 Logistic Regression", "Gradient-Boosted Trees"]
_BAR_COLORS = [_TXT2, _COLS[0], _COLS[1], _COLS[2]]

fig_model_comparison, _comp_axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=_BG)

for _idx, _target in enumerate(_TARGETS):
    _ax = _comp_axes[_idx]
    _ax.set_facecolor(_BG)
    
    _res_df = model_comparison_results[_target]
    _is_primary = " ★" if _target == primary_objective else ""
    
    _n_models = len(_ALL_MODELS)
    _n_metrics = len(_METRICS)
    _x = np.arange(_n_metrics)
    _width = 0.18
    
    for _m_idx, _model_name in enumerate(_ALL_MODELS):
        _row = _res_df[_res_df["Model"] == _model_name].iloc[0]
        _vals = [_row[_met] for _met in _METRICS]
        _offset = (_m_idx - (_n_models - 1) / 2) * _width
        _bars = _ax.bar(_x + _offset, _vals, _width, label=_model_name,
                        color=_BAR_COLORS[_m_idx], edgecolor="none", alpha=0.9)
    
    _ax.set_title(f"{_TARGET_LABELS[_target]}{_is_primary}", color=_TXT, fontsize=13, fontweight="bold", pad=10)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_METRICS, fontsize=10, rotation=15)
    _ax.tick_params(colors=_TXT2, labelsize=9)
    if _idx == 0:
        _ax.set_ylabel("Score", color=_TXT, fontsize=11)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["bottom"].set_color(_TXT2)
    _ax.spines["left"].set_color(_TXT2)
    
    if _idx == 2:
        _ax.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=8,
                   loc="upper right", bbox_to_anchor=(1.0, 1.0))

plt.suptitle("Model Performance Comparison (Test Set)", color=_TXT, fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# ═════════════════════════════════════════════════════
# 3. DELTA HEATMAP — improvement over constant baseline
# ═════════════════════════════════════════════════════
_delta_metrics = ["ΔROC-AUC", "ΔPR-AUC", "ΔLift@10%", "ΔLift@20%"]
_model_names_short = ["LR (active_days)", "L2 Logistic Reg.", "Gradient-Boosted Trees"]

fig_delta_heatmap, _hm_axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=_BG)

for _idx, _target in enumerate(_TARGETS):
    _ax = _hm_axes[_idx]
    _ax.set_facecolor(_BG)
    
    _dt = model_delta_tables[_target]
    _non_baseline = _dt[_dt["Model"] != "Constant (base rate)"]
    
    _data = np.zeros((len(_model_names_short), len(_delta_metrics)))
    for _r_idx, (_, _row) in enumerate(_non_baseline.iterrows()):
        for _c_idx, _dm in enumerate(_delta_metrics):
            _data[_r_idx, _c_idx] = _row[_dm]
    
    _im = _ax.imshow(_data, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=max(3.5, _data.max() * 1.1))
    
    # Annotate cells
    for _r in range(len(_model_names_short)):
        for _c in range(len(_delta_metrics)):
            _val = _data[_r, _c]
            _txt_color = _BG if abs(_val) > 0.5 else _TXT
            _ax.text(_c, _r, f"{_val:+.2f}", ha="center", va="center",
                     fontsize=10, fontweight="bold", color=_txt_color)
    
    _ax.set_xticks(range(len(_delta_metrics)))
    _ax.set_xticklabels(["ΔROC", "ΔPR-AUC", "ΔL@10%", "ΔL@20%"], fontsize=9, rotation=20)
    _ax.set_yticks(range(len(_model_names_short)))
    if _idx == 0:
        _ax.set_yticklabels(_model_names_short, fontsize=9)
    else:
        _ax.set_yticklabels([])
    _ax.tick_params(colors=_TXT2, labelsize=9)
    
    _is_primary = " ★" if _target == primary_objective else ""
    _ax.set_title(f"{_TARGET_LABELS[_target]}{_is_primary}", color=_TXT, fontsize=12, fontweight="bold", pad=8)

plt.suptitle("Improvement Over Constant Baseline (Δ Metrics)", color=_TXT, fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# ═════════════════════════════════════════════════════
# 4. SUMMARY
# ═════════════════════════════════════════════════════
print("=" * 65)
print("🏆 FINAL MODEL EVALUATION SUMMARY")
print("=" * 65)
print(f"\n   Primary Objective: {primary_objective} ({_TARGET_LABELS[primary_objective]})")
print(f"   Fallback Triggered: {fallback_triggered}")

_primary_results = model_comparison_results[primary_objective]
_best_row = _primary_results.loc[_primary_results["PR-AUC"].idxmax()]
print(f"\n   Best Model (by PR-AUC): {_best_row['Model']}")
print(f"      ROC-AUC: {_best_row['ROC-AUC']:.4f}")
print(f"      PR-AUC:  {_best_row['PR-AUC']:.4f}")
print(f"      Lift@10%: {_best_row['Lift@10%']:.2f}")
print(f"      Lift@20%: {_best_row['Lift@20%']:.2f}")

print(f"\n✅ Calibration plots, comparison charts, and delta heatmaps generated.")
