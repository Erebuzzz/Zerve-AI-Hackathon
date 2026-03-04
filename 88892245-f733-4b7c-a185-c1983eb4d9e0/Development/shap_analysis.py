
import subprocess
import sys

# Install to /tmp which is writable
_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "shap", "--target", "/tmp/shap_libs", "-q"],
    capture_output=True, text=True
)
print("Install result:", _result.returncode)
if _result.returncode != 0:
    print("Error:", _result.stderr[-300:])

# Add to path
sys.path.insert(0, "/tmp/shap_libs")

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════
# Zerve Design System
# ═════════════════════════════════════════════════════
_BG = "#1D1D20"
_TXT = "#fbfbff"
_TXT2 = "#909094"
_COLS = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
         "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
_HL = "#ffd400"

# ═════════════════════════════════════════════════════
# 0. REBUILD FEATURE DATA & MODELS FROM UPSTREAM
# ═════════════════════════════════════════════════════
_TARGETS = ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]
_META_COLS = ["user_id_canon", "split"] + _TARGETS
_FEATURE_COLS = [c for c in modeling_df.columns if c not in _META_COLS]

def _clean_name(s):
    return s.replace("feat_", "").replace("_", " ").title()

_CLEAN_NAMES = {c: _clean_name(c) for c in _FEATURE_COLS}
_X_all = modeling_df[_FEATURE_COLS].values

# ═════════════════════════════════════════════════════
# 1. COMPUTE SHAP FOR GBT MODELS (all 3 targets)
# ═════════════════════════════════════════════════════
shap_results = {}

for _target in _TARGETS:
    _gbt_model = all_models[(_target, "GBT")]
    _explainer = shap.TreeExplainer(_gbt_model)
    _shap_values = _explainer.shap_values(_X_all)
    shap_results[_target] = {
        "shap_values": _shap_values,
        "expected_value": _explainer.expected_value,
    }
    print(f"✅ SHAP computed for {_target}: shape={_shap_values.shape}")

# ═════════════════════════════════════════════════════
# 2. TOP-20 SHAP FEATURE IMPORTANCE RANKING
# ═════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📊 TOP-20 FEATURE IMPORTANCE BY MEAN |SHAP| VALUE")
print("=" * 70)

shap_importance_tables = {}

for _target in _TARGETS:
    _sv = shap_results[_target]["shap_values"]
    _mean_abs = np.abs(_sv).mean(axis=0)
    _rank_idx = np.argsort(_mean_abs)[::-1][:20]
    
    _rows = []
    for _rank, _i in enumerate(_rank_idx):
        _rows.append({
            "Rank": _rank + 1,
            "Feature": _FEATURE_COLS[_i],
            "Clean Name": _CLEAN_NAMES[_FEATURE_COLS[_i]],
            "Mean |SHAP|": _mean_abs[_i],
        })
    
    _imp_df = pd.DataFrame(_rows)
    shap_importance_tables[_target] = _imp_df
    
    _lbl = {"y_ret_30d": "30-Day Retention", "y_ret_90d": "90-Day Retention", "y_upgrade_60d": "60-Day Upgrade"}[_target]
    _is_primary = " ★ PRIMARY" if _target == primary_objective else ""
    print(f"\n🎯 {_lbl}{_is_primary}")
    for _, _r in _imp_df.head(20).iterrows():
        _bar = "█" * int(_r["Mean |SHAP|"] / _imp_df["Mean |SHAP|"].max() * 30)
        print(f"   {_r['Rank']:2d}. {_r['Clean Name']:40s} {_r['Mean |SHAP|']:.4f}  {_bar}")

# ═════════════════════════════════════════════════════
# 3. SHAP SUMMARY PLOTS (beeswarm) — all 3 targets
# ═════════════════════════════════════════════════════
for _target in _TARGETS:
    _sv = shap_results[_target]["shap_values"]
    _lbl = {"y_ret_30d": "30-Day Retention", "y_ret_90d": "90-Day Retention", "y_upgrade_60d": "60-Day Upgrade"}[_target]
    _is_primary = " ★" if _target == primary_objective else ""
    
    _mean_abs = np.abs(_sv).mean(axis=0)
    _top_idx = np.argsort(_mean_abs)[::-1][:15]
    _top_names = [_CLEAN_NAMES[_FEATURE_COLS[i]] for i in _top_idx]
    _top_shap = _sv[:, _top_idx]
    _top_feat_vals = _X_all[:, _top_idx]
    
    _fig, _ax = plt.subplots(figsize=(12, 8), facecolor=_BG)
    _ax.set_facecolor(_BG)
    
    for _feat_rank in range(len(_top_idx)):
        _shap_vals = _top_shap[:, _feat_rank]
        _feat_vals = _top_feat_vals[:, _feat_rank]
        
        _fmin, _fmax = _feat_vals.min(), _feat_vals.max()
        if _fmax > _fmin:
            _norm_vals = (_feat_vals - _fmin) / (_fmax - _fmin)
        else:
            _norm_vals = np.full_like(_feat_vals, 0.5)
        
        _y_pos = len(_top_idx) - 1 - _feat_rank
        _jitter = np.random.RandomState(42).uniform(-0.3, 0.3, len(_shap_vals))
        _colors = plt.cm.coolwarm(_norm_vals)
        _ax.scatter(_shap_vals, _y_pos + _jitter, c=_colors, s=4, alpha=0.5, edgecolors="none")
    
    _ax.set_yticks(range(len(_top_idx)))
    _ax.set_yticklabels(list(reversed(_top_names)), fontsize=10, color=_TXT)
    _ax.set_xlabel("SHAP Value (impact on model output)", color=_TXT, fontsize=12)
    _ax.axvline(0, color=_TXT2, linewidth=0.8, linestyle="--", alpha=0.5)
    _ax.set_title(f"SHAP Summary — {_lbl}{_is_primary}", color=_TXT, fontsize=14, fontweight="bold", pad=15)
    _ax.tick_params(colors=_TXT2, labelsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["bottom"].set_color(_TXT2)
    _ax.spines["left"].set_color(_TXT2)
    
    _sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    _sm.set_array([])
    _cbar = plt.colorbar(_sm, ax=_ax, fraction=0.03, pad=0.02)
    _cbar.set_label("Feature Value (normalized)", color=_TXT, fontsize=10)
    _cbar.ax.tick_params(labelcolor=_TXT2, labelsize=8)
    _cbar.set_ticks([0, 0.5, 1])
    _cbar.set_ticklabels(["Low", "Mid", "High"])
    
    plt.tight_layout()
    plt.show()
    
    if _target == "y_ret_30d":
        fig_shap_summary_ret30 = _fig
    elif _target == "y_ret_90d":
        fig_shap_summary_ret90 = _fig
    else:
        fig_shap_summary_upgrade = _fig

# ═════════════════════════════════════════════════════
# 4. SHAP DEPENDENCE PLOTS — top 4 features for primary objective
# ═════════════════════════════════════════════════════
_primary_sv = shap_results[primary_objective]["shap_values"]
_primary_mean = np.abs(_primary_sv).mean(axis=0)
_primary_top_idx = np.argsort(_primary_mean)[::-1][:4]

fig_shap_dependence, _dep_axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=_BG)
_dep_axes = _dep_axes.flatten()

_primary_lbl = {"y_ret_30d": "30-Day Retention", "y_ret_90d": "90-Day Retention", "y_upgrade_60d": "60-Day Upgrade"}[primary_objective]

for _plot_idx, _feat_idx in enumerate(_primary_top_idx):
    _ax = _dep_axes[_plot_idx]
    _ax.set_facecolor(_BG)
    
    _feat_vals = _X_all[:, _feat_idx]
    _shap_vals = _primary_sv[:, _feat_idx]
    _feat_name = _CLEAN_NAMES[_FEATURE_COLS[_feat_idx]]
    
    _corrs = []
    for _other_idx in range(len(_FEATURE_COLS)):
        if _other_idx != _feat_idx:
            _c = np.abs(np.corrcoef(_X_all[:, _other_idx], _shap_vals)[0, 1])
            _corrs.append((_other_idx, _c if not np.isnan(_c) else 0))
    _corrs.sort(key=lambda x: x[1], reverse=True)
    _interact_idx = _corrs[0][0]
    _interact_vals = _X_all[:, _interact_idx]
    _interact_name = _CLEAN_NAMES[_FEATURE_COLS[_interact_idx]]
    
    _imin, _imax = _interact_vals.min(), _interact_vals.max()
    if _imax > _imin:
        _norm_interact = (_interact_vals - _imin) / (_imax - _imin)
    else:
        _norm_interact = np.full_like(_interact_vals, 0.5)
    
    _colors = plt.cm.coolwarm(_norm_interact)
    
    _ax.scatter(_feat_vals, _shap_vals, c=_colors, s=8, alpha=0.6, edgecolors="none")
    _ax.axhline(0, color=_TXT2, linewidth=0.8, linestyle="--", alpha=0.5)
    _ax.set_xlabel(_feat_name, color=_TXT, fontsize=11)
    _ax.set_ylabel("SHAP Value", color=_TXT, fontsize=11)
    _ax.set_title(f"{_feat_name}\n(colored by {_interact_name})", color=_TXT, fontsize=11, fontweight="bold", pad=8)
    _ax.tick_params(colors=_TXT2, labelsize=9)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["bottom"].set_color(_TXT2)
    _ax.spines["left"].set_color(_TXT2)

plt.suptitle(f"SHAP Dependence Plots — {_primary_lbl} (GBT)", color=_TXT, fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# ═════════════════════════════════════════════════════
# 5. SIGNAL STRENGTH CLASSIFICATION
# ═════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🔬 SIGNAL STRENGTH CLASSIFICATION")
print("=" * 70)

_top10_sets = {}
for _target in _TARGETS:
    _sv = shap_results[_target]["shap_values"]
    _mean_abs = np.abs(_sv).mean(axis=0)
    _top10_idx = np.argsort(_mean_abs)[::-1][:10]
    _top10_sets[_target] = set(_FEATURE_COLS[i] for i in _top10_idx)

_all_top10_features = set()
for _s in _top10_sets.values():
    _all_top10_features |= _s

print("\n🟢 STRONG SIGNALS (top-10 in ≥2 targets):")
_strong_features = []
_tentative_features = []

for _f in sorted(_all_top10_features):
    _count = sum(1 for _t in _TARGETS if _f in _top10_sets[_t])
    _targets_in = [t for t in _TARGETS if _f in _top10_sets[t]]
    if _count >= 2:
        _strong_features.append(_f)
        print(f"   ✅ {_CLEAN_NAMES[_f]:40s} (in {_count}/3 targets: {', '.join(_targets_in)})")
    else:
        _tentative_features.append(_f)

print("\n🟡 TENTATIVE SIGNALS (top-10 in only 1 target):")
for _f in sorted(_tentative_features):
    _targets_in = [t for t in _TARGETS if _f in _top10_sets[t]]
    print(f"   ⚠️  {_CLEAN_NAMES[_f]:40s} (in: {', '.join(_targets_in)})")

print(f"\n📊 Summary: {len(_strong_features)} strong, {len(_tentative_features)} tentative signals")
print("✅ SHAP analysis complete.")
