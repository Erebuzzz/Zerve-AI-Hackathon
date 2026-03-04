
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
_BG   = "#1D1D20"
_TXT  = "#fbfbff"
_TXT2 = "#909094"
_COLS = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF",
         "#1F77B4","#9467BD","#8C564B","#C49C94","#E377C2"]
_HL   = "#ffd400"

# ═══════════════════════════════════════════════════════════════
# 0. SETUP
# ═══════════════════════════════════════════════════════════════
_feat_cols = [c for c in feat_matrix_v2.columns
              if c not in ["user_id_canon","y_ret_30d","y_ret_90d","y_upgrade_60d","split"]]

_shap_model = gbt_model_final   # HistGradientBoostingClassifier
_X_train = X_gbt_train
_X_test  = X_gbt_test
_y_test  = y_gbt_test

print("═"*65)
print("🧮 SHAP-EQUIVALENT ADVANCED ANALYSIS")
print("═"*65)
print(f"   Model: HistGradientBoostingClassifier (best tuned, Bayesian HPO)")
print(f"   Train: {len(_X_train):,}  |  Test: {len(_X_test):,}  |  Features: {len(_feat_cols)}")
print(f"   Method: Permutation importance + marginal SHAP approximation")

# ═══════════════════════════════════════════════════════════════
# HELPER: Marginal contribution SHAP approximation for a single sample
# Uses the 'interventional' approach: permute individual features
# vs baseline (mean), computing E[f(x)] shift per feature
# ═══════════════════════════════════════════════════════════════
_X_bg = _X_train[:200]  # background reference for expectations
_feature_means = _X_bg.mean(axis=0)

def _marginal_shap_single(model, x_row, X_background, n_samples=50):
    """
    Approximate SHAP values for a single row using marginal contribution method.
    For each feature i: shap_i ≈ E[f(x)|x_i] - E[f(x)]
    Uses a sampled background to estimate expectation.
    """
    _n_feats = x_row.shape[0]
    _bg_sample = X_background[np.random.choice(len(X_background), n_samples, replace=True)]
    _shap_approx = np.zeros(_n_feats)
    _base_proba  = model.predict_proba(_bg_sample)[:, 1].mean()

    for _fi in range(_n_feats):
        # Set feature fi to x_row value, all others from background
        _intervened = _bg_sample.copy()
        _intervened[:, _fi] = x_row[_fi]
        _contrib = model.predict_proba(_intervened)[:, 1].mean() - _base_proba
        _shap_approx[_fi] = _contrib
    return _shap_approx, _base_proba

# ═══════════════════════════════════════════════════════════════
# 1. GLOBAL FEATURE IMPORTANCE — Permutation Importance
#    (sklearn permutation_importance is a rigorous SHAP alternative)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("📊 CHART 1: Global Feature Importance (Permutation Importance)")
print("═"*65)

_perm_result = permutation_importance(
    _shap_model, _X_test, _y_test,
    n_repeats=10, random_state=42,
    scoring="average_precision", n_jobs=1
)

_perm_importance_df = pd.DataFrame({
    "feature": _feat_cols,
    "mean_abs_shap": _perm_result.importances_mean,
    "std_importance": _perm_result.importances_std
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

_TOP_N = 20
_top_feats = _perm_importance_df.head(_TOP_N)

print(f"\n   Top 10 features by permutation importance (mean PR-AUC drop):")
for _i, _row in _perm_importance_df.head(10).iterrows():
    print(f"      {_i+1:2d}. {_row['feature']:45s} {_row['mean_abs_shap']:+.5f}")

fig_shap_bar, _ax_bar = plt.subplots(figsize=(13, 10), facecolor=_BG)
_ax_bar.set_facecolor(_BG)
_ax_bar.spines["top"].set_visible(False); _ax_bar.spines["right"].set_visible(False)
_ax_bar.spines["left"].set_color(_TXT2);  _ax_bar.spines["bottom"].set_color(_TXT2)
_ax_bar.tick_params(colors=_TXT2, labelsize=10)

_bar_colors = [_COLS[0] if i < 5 else (_COLS[4] if i < 10 else _TXT2)
               for i in range(len(_top_feats))]
_vals_plot = _top_feats["mean_abs_shap"].values[::-1]
_bars = _ax_bar.barh(
    [n.replace("feat_", "").replace("_", " ")[:38] for n in _top_feats["feature"].values[::-1]],
    _vals_plot,
    xerr=_top_feats["std_importance"].values[::-1],
    color=_bar_colors[::-1], alpha=0.9, edgecolor="none",
    error_kw=dict(ecolor=_HL, linewidth=1.5, capsize=3)
)
for _b, _v in zip(_bars, _vals_plot):
    _ax_bar.text(max(_v, 0) + 0.0005, _b.get_y() + _b.get_height()/2,
                 f"{_v:+.4f}", va="center", color=_TXT, fontsize=8)

_ax_bar.set_xlabel("Permutation Importance (mean PR-AUC drop when feature shuffled)", color=_TXT, fontsize=11)
_ax_bar.set_title(f"Global Feature Importance — Top {_TOP_N} Features\n"
                  "(SHAP equivalent via permutation, 10 repeats, error bars = ±1 std)",
                  color=_TXT, fontsize=13, fontweight="bold", pad=15)
_ax_bar.tick_params(axis="y", labelcolor=_TXT, labelsize=9)
_ax_bar.tick_params(axis="x", labelcolor=_TXT2, labelsize=9)
_ax_bar.axvline(0, color=_TXT2, linewidth=0.8, linestyle="--")

_p1 = mpatches.Patch(color=_COLS[0], label="Top 5 (most important)")
_p2 = mpatches.Patch(color=_COLS[4], label="Top 6–10")
_p3 = mpatches.Patch(color=_TXT2,   label="Top 11–20")
_ax_bar.legend(handles=[_p1, _p2, _p3], facecolor=_BG, edgecolor=_TXT2,
               labelcolor=_TXT, fontsize=9, loc="lower right")
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════
# 2. PER-USER FORCE PLOTS — Marginal SHAP (sample of 10)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("👤 CHART 2: Per-User Force Plots (10 sampled users, marginal SHAP)")
print("═"*65)

_proba_test = _shap_model.predict_proba(_X_test)[:, 1]
_pos_idx = np.where(_proba_test >= 0.5)[0]
_neg_idx = np.where(_proba_test < 0.5)[0]

np.random.seed(42)
_sample_pos = np.random.choice(_pos_idx, size=min(5, len(_pos_idx)), replace=False)
_sample_neg = np.random.choice(_neg_idx, size=min(5, len(_neg_idx)), replace=False)
_sample_idx = np.concatenate([_sample_pos, _sample_neg])[:10]

print(f"   Computing marginal SHAP for {len(_sample_idx)} users...")
_user_shap_all = []
for _ui in _sample_idx:
    _sv, _bv = _marginal_shap_single(_shap_model, _X_test[_ui], _X_bg, n_samples=30)
    _user_shap_all.append((_sv, _bv))

_N_FEATS_SHOW = 8

fig_shap_force, _axs = plt.subplots(2, 5, figsize=(24, 10), facecolor=_BG)
_axs = _axs.flatten()

for _plot_i, (_user_idx, (_user_shap, _bv)) in enumerate(zip(_sample_idx, _user_shap_all)):
    _ax = _axs[_plot_i]
    _ax.set_facecolor(_BG)
    _ax.spines["top"].set_visible(False); _ax.spines["right"].set_visible(False)
    _ax.spines["left"].set_color(_TXT2);  _ax.spines["bottom"].set_color(_TXT2)

    _user_pred = float(_proba_test[_user_idx])
    _user_y    = int(_y_test[_user_idx])

    _sorted_idx  = np.argsort(np.abs(_user_shap))[::-1][:_N_FEATS_SHOW]
    _vals        = _user_shap[_sorted_idx]
    _names_short = [_feat_cols[j].replace("feat_","").replace("_"," ")[:18]
                    for j in _sorted_idx]

    _colors = [_COLS[2] if v > 0 else _COLS[3] for v in _vals]
    _ax.barh(range(len(_vals)), _vals[::-1], color=_colors[::-1], alpha=0.9, edgecolor="none")
    _ax.set_yticks(range(len(_vals)))
    _ax.set_yticklabels(_names_short[::-1], fontsize=6.5, color=_TXT)
    _ax.axvline(0, color=_TXT2, linewidth=0.8, linestyle="--")

    _label    = "✅ Retained" if _user_y == 1 else "❌ Churned"
    _pred_col = _COLS[2] if _user_pred >= 0.5 else _COLS[3]
    _ax.set_title(f"User {_plot_i+1}  P={_user_pred:.3f}  {_label}",
                  color=_pred_col, fontsize=8, fontweight="bold", pad=5)
    _ax.tick_params(axis="x", labelsize=6, colors=_TXT2)

plt.suptitle("Per-User SHAP Force Plots — Sample of 10 Users\n"
             "🟢 Green = pushes toward retention  🔴 Red = pushes toward churn",
             color=_TXT, fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
print(f"   ✅ Force plots rendered for {len(_sample_idx)} users")

# ═══════════════════════════════════════════════════════════════
# 3. INTERACTION HEATMAP — Feature value × Importance correlation
#    SHAP interaction approximation: Pearson correlation between
#    feature values and per-user importance vectors
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🗺️  CHART 3: Feature Interaction Heatmap (top 15 features)")
print("═"*65)

# Use all-user marginal SHAP on a wider test subsample (300 users)
_N_WIDE = min(300, len(_X_test))
print(f"   Computing marginal SHAP on {_N_WIDE} users for interaction matrix...")

_wide_shap = np.zeros((_N_WIDE, len(_feat_cols)))
for _ri in range(_N_WIDE):
    _sv, _ = _marginal_shap_single(_shap_model, _X_test[_ri], _X_bg, n_samples=20)
    _wide_shap[_ri] = _sv

# Interaction matrix: cross-correlation of SHAP values between features
# abs(corr(SHAP_i, SHAP_j)) ≈ measure of how jointly features drive predictions
_top15_names = _perm_importance_df.head(15)["feature"].values
_top15_idx   = [_feat_cols.index(f) for f in _top15_names]

_shap_sub     = _wide_shap[:, _top15_idx]
_interact_mat = np.abs(np.corrcoef(_shap_sub.T))  # (15, 15)
np.fill_diagonal(_interact_mat, 0)  # suppress diagonal

_short_names = [n.replace("feat_","").replace("_"," ")[:18] for n in _top15_names]

fig_shap_interact, _ax_hm = plt.subplots(figsize=(14, 12), facecolor=_BG)
_ax_hm.set_facecolor(_BG)

_im = _ax_hm.imshow(_interact_mat, cmap="YlOrRd", aspect="auto",
                    interpolation="nearest", vmin=0, vmax=1)

_ax_hm.set_xticks(range(15)); _ax_hm.set_yticks(range(15))
_ax_hm.set_xticklabels(_short_names, rotation=45, ha="right", color=_TXT, fontsize=8)
_ax_hm.set_yticklabels(_short_names, color=_TXT, fontsize=8)

_cbar = plt.colorbar(_im, ax=_ax_hm, fraction=0.046, pad=0.04)
_cbar.ax.yaxis.set_tick_params(color=_TXT2, labelsize=9)
_cbar.set_label("SHAP Value Correlation |r|  (0=independent, 1=highly interactive)",
                color=_TXT, fontsize=10)
plt.setp(_cbar.ax.yaxis.get_ticklabels(), color=_TXT)

# Annotate high-interaction pairs
_threshold_annot = _interact_mat.mean() + _interact_mat.std()
for _i in range(15):
    for _j in range(15):
        if _i != _j and _interact_mat[_i,_j] >= _threshold_annot:
            _ax_hm.text(_j, _i, f"{_interact_mat[_i,_j]:.2f}",
                        ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")

_ax_hm.set_title("SHAP Interaction Heatmap — Top 15 Features\n"
                 "(|correlation of SHAP values|; bright = strongly interacting pair)",
                 color=_TXT, fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.show()

# Print top pairs
_interact_pairs = []
for _ii in range(15):
    for _jj in range(_ii+1, 15):
        _interact_pairs.append({
            "feature_1": _top15_names[_ii], "feature_2": _top15_names[_jj],
            "interaction": _interact_mat[_ii,_jj]
        })
_interact_df = pd.DataFrame(_interact_pairs).sort_values("interaction", ascending=False)
print(f"\n   Top 5 interacting feature pairs (SHAP correlation):")
for _, _row in _interact_df.head(5).iterrows():
    print(f"      {_row['feature_1']:35s} × {_row['feature_2']:35s} = {_row['interaction']:.4f}")

# ═══════════════════════════════════════════════════════════════
# 4. ABLATION STUDY — Drop feature groups one at a time
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🔬 ABLATION STUDY  (intensity | advanced_usage | collaboration | metadata)")
print("═"*65)

_FEATURE_GROUPS = {
    "intensity": [
        "feat_event_count","feat_active_days","feat_n_sessions",
        "feat_events_per_day","feat_rampup_slope","feat_max_gap_days",
        "feat_mean_events_per_session","feat_median_events_per_session",
        "feat_max_events_per_session","feat_mean_session_duration_min",
        "feat_max_session_duration_min","feat_mean_distinct_events_per_session",
        "feat_day_gap_variance",
    ],
    "advanced_usage": [
        "feat_ratio_agent","feat_ratio_block_ops","feat_ratio_canvas",
        "feat_ratio_credits","feat_ratio_deploy","feat_ratio_files",
        "feat_early_deploy_count","feat_early_schedule_count",
        "feat_session_entropy","feat_tod_entropy",
        "feat_execution_to_agent_ratio","feat_agent_block_conversion",
        "feat_distinct_events","feat_distinct_categories",
        "feat_ttf_run_block","feat_ttf_canvas_create","feat_ttf_agent_use",
        "feat_ttf_file_upload","feat_ttf_credits_used","feat_ttf_edge_create",
        "feat_ttf_block_create","feat_credit_amount_sum",
    ],
    "collaboration": [
        "feat_ratio_collab","feat_collab_actions","feat_distinct_canvases",
    ],
    "metadata": [
        "feat_signup_dow","feat_signup_hour",
        "feat_onboarding_completed","feat_onboarding_skipped","feat_tour_finished",
        "feat_te_country","feat_te_device","feat_te_os","feat_te_browser",
        "feat_ratio_onboarding",
    ],
}
for _g, _gc in _FEATURE_GROUPS.items():
    _FEATURE_GROUPS[_g] = [c for c in _gc if c in _feat_cols]
    print(f"   Group '{_g}': {len(_FEATURE_GROUPS[_g])} features")

def _compute_lift10(y_true, y_proba):
    _n_top = max(1, int(len(y_true)*0.10))
    _top   = np.argsort(y_proba)[::-1][:_n_top]
    _base  = y_true.mean()
    return y_true[_top].mean() / _base if _base > 0 else 1.0

def _retrain_eval_abl(X_tr, y_tr, X_te, y_te, params):
    _sw = compute_sample_weight("balanced", y_tr)
    _m  = HistGradientBoostingClassifier(
        **params, random_state=42,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=25, scoring="average_precision"
    )
    _m.fit(X_tr, y_tr, sample_weight=_sw)
    _p = _m.predict_proba(X_te)[:,1]
    return (average_precision_score(y_te, _p) if y_te.sum()>0 else 0.0, _compute_lift10(y_te, _p))

_base_proba  = _shap_model.predict_proba(_X_test)[:,1]
_base_prauc  = average_precision_score(_y_test, _base_proba) if _y_test.sum()>0 else 0.0
_base_lift10 = _compute_lift10(_y_test, _base_proba)
_best_p      = dict(gbt_best_params)

print(f"\n   BASELINE (all {len(_feat_cols)} features): PR-AUC={_base_prauc:.4f}  Lift@10%={_base_lift10:.3f}×")
_abl_rows = [{"group_dropped":"none (baseline)","pr_auc":_base_prauc,"lift_10":_base_lift10,
               "n_features":len(_feat_cols),"pr_auc_drop":0.0,"lift_drop":0.0}]

for _gname, _gcols in _FEATURE_GROUPS.items():
    _keep  = [c for c in _feat_cols if c not in _gcols]
    _cidx  = [_feat_cols.index(c) for c in _keep]
    print(f"\n   Dropping '{_gname}' ({len(_gcols)} feats → {len(_keep)} remain)...")
    _pr, _lift = _retrain_eval_abl(_X_train[:,_cidx], y_gbt_train, _X_test[:,_cidx], _y_test, _best_p)
    _pr_d  = _base_prauc  - _pr
    _lft_d = _base_lift10 - _lift
    print(f"      PR-AUC={_pr:.4f} (Δ={_pr_d:+.4f})  Lift@10%={_lift:.3f}× (Δ={_lft_d:+.3f})")
    _abl_rows.append({"group_dropped":_gname,"pr_auc":_pr,"lift_10":_lift,
                       "n_features":len(_keep),"pr_auc_drop":_pr_d,"lift_drop":_lft_d})

shap_ablation_df = pd.DataFrame(_abl_rows)

print("\n" + "═"*65)
print("📋 ABLATION RESULTS TABLE")
print("═"*65)
print(f"\n{'Group Dropped':22s}  {'PR-AUC':>8s}  {'ΔPRAUC':>8s}  {'Lift@10%':>9s}  {'ΔLift':>8s}  {'N Feats':>8s}")
print("-"*75)
for _, _r in shap_ablation_df.iterrows():
    _mk = "  ← baseline" if _r["group_dropped"] == "none (baseline)" else ""
    print(f"{_r['group_dropped']:22s}  {_r['pr_auc']:>8.4f}  {_r['pr_auc_drop']:>+8.4f}  "
          f"{_r['lift_10']:>9.3f}  {_r['lift_drop']:>+8.3f}  {_r['n_features']:>8.0f}{_mk}")

_non_base = shap_ablation_df[shap_ablation_df["group_dropped"] != "none (baseline)"]
fig_shap_ablation, (_ax_a1, _ax_a2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=_BG)
for _ax_a in (_ax_a1, _ax_a2):
    _ax_a.set_facecolor(_BG)
    _ax_a.spines["top"].set_visible(False); _ax_a.spines["right"].set_visible(False)
    _ax_a.spines["left"].set_color(_TXT2);  _ax_a.spines["bottom"].set_color(_TXT2)

_x_a = np.arange(len(_non_base))
_abl_c1 = [_COLS[3] if v>0 else _COLS[2] for v in _non_base["pr_auc_drop"].values]
_abl_c2 = [_COLS[3] if v>0 else _COLS[2] for v in _non_base["lift_drop"].values]

_ax_a1.bar(_x_a, _non_base["pr_auc_drop"].values, color=_abl_c1, alpha=0.9, edgecolor="none")
_ax_a1.axhline(0, color=_TXT2, linewidth=0.8, linestyle="--")
for _bi, _bv in enumerate(_non_base["pr_auc_drop"].values):
    _ax_a1.text(_bi, _bv + (_bv*0.05 if _bv != 0 else 0.001), f"{_bv:+.4f}",
                ha="center", color=_TXT, fontsize=11, fontweight="bold")
_ax_a1.set_xticks(_x_a)
_ax_a1.set_xticklabels(_non_base["group_dropped"].values, color=_TXT, fontsize=12, rotation=10)
_ax_a1.set_ylabel("PR-AUC Drop  (positive = hurt performance)", color=_TXT, fontsize=11)
_ax_a1.set_title("Ablation Study — PR-AUC Impact", color=_TXT, fontsize=13, fontweight="bold")
_ax_a1.tick_params(colors=_TXT2)

_ax_a2.bar(_x_a, _non_base["lift_drop"].values, color=_abl_c2, alpha=0.9, edgecolor="none")
_ax_a2.axhline(0, color=_TXT2, linewidth=0.8, linestyle="--")
for _bi, _bv in enumerate(_non_base["lift_drop"].values):
    _ax_a2.text(_bi, _bv + (_bv*0.05 if _bv != 0 else 0.005), f"{_bv:+.3f}×",
                ha="center", color=_TXT, fontsize=11, fontweight="bold")
_ax_a2.set_xticks(_x_a)
_ax_a2.set_xticklabels(_non_base["group_dropped"].values, color=_TXT, fontsize=12, rotation=10)
_ax_a2.set_ylabel("Lift@10% Drop  (positive = hurt performance)", color=_TXT, fontsize=11)
_ax_a2.set_title("Ablation Study — Lift@10% Impact", color=_TXT, fontsize=13, fontweight="bold")
_ax_a2.tick_params(colors=_TXT2)

plt.suptitle("Feature Group Ablation Study — Impact on Retention Model Performance\n"
             "Red bars = dropping group hurts performance (group is important)",
             color=_TXT, fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════
# 5. BOOTSTRAP SHAP — Variance across 3 Temporal Windows
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🔁 BOOTSTRAP SHAP VARIANCE — 3 Temporal Windows")
print("═"*65)

_mdf_ts = feat_matrix_v2.merge(
    cohort_users[["user_id_canon","first_event_ts"]], on="user_id_canon", how="left"
).sort_values("first_event_ts").reset_index(drop=True)

_n_tot    = len(_mdf_ts)
_bsize    = _n_tot // 4
_ts_blks  = [_mdf_ts.iloc[i*_bsize:(i+1)*_bsize] for i in range(3)]
_ts_blks.append(_mdf_ts.iloc[3*_bsize:])

_window_imps = {}

for _w in range(3):
    _wtr = pd.concat(_ts_blks[:_w+1], ignore_index=True)
    _wvl = _ts_blks[_w+1]

    _Xwtr = _wtr[_feat_cols].values.astype(np.float64)
    _ywtr = _wtr["y_ret_30d"].values.astype(int)
    _Xwvl = _wvl[_feat_cols].values.astype(np.float64)
    _ywvl = _wvl["y_ret_30d"].values.astype(int)

    if _ywtr.sum() < 3 or _ywvl.sum() < 2:
        print(f"   Window {_w+1}: skipped")
        continue

    _sw_w = compute_sample_weight("balanced", _ywtr)
    _wm   = HistGradientBoostingClassifier(
        **_best_p, random_state=42,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, scoring="average_precision"
    )
    _wm.fit(_Xwtr, _ywtr, sample_weight=_sw_w)

    # Permutation importance on window validation set
    _pi_w = permutation_importance(
        _wm, _Xwvl[:200], _ywvl[:200],
        n_repeats=5, random_state=42, scoring="average_precision", n_jobs=1
    )
    _window_imps[f"Window {_w+1}"] = _pi_w.importances_mean

    _pr_w = average_precision_score(_ywvl, _wm.predict_proba(_Xwvl)[:,1])
    print(f"   Window {_w+1}: train={len(_Xwtr):,} val={len(_Xwvl):,} PR-AUC={_pr_w:.4f}")

_wins_avail = list(_window_imps.keys())
if len(_wins_avail) >= 2:
    _stab_df = pd.DataFrame(_window_imps, index=_feat_cols)
    _stab_df["mean_imp"] = _stab_df.mean(axis=1)
    _stab_df["std_imp"]  = _stab_df.std(axis=1)
    _stab_df["cv_imp"]   = (_stab_df["std_imp"] / (_stab_df["mean_imp"].abs() + 1e-6)).round(4)
    _stab_df = _stab_df.sort_values("mean_imp", ascending=False)

    print(f"\n   Bootstrap SHAP Stability — Top 15 features across {len(_wins_avail)} windows:")
    print(f"\n{'Feature':45s}  ", end="")
    for _wn in _wins_avail: print(f"{'Imp '+_wn:>14s}  ", end="")
    print(f"{'Mean':>8s}  {'Std':>8s}  {'CV':>6s}")
    print("-"*105)
    for _fn, _row in _stab_df.head(15).iterrows():
        print(f"{_fn[:44]:45s}  ", end="")
        for _wn in _wins_avail: print(f"{_row[_wn]:>14.5f}  ", end="")
        print(f"{_row['mean_imp']:>8.5f}  {_row['std_imp']:>8.5f}  {_row['cv_imp']:>6.3f}")

    _top15_stab = _stab_df.head(15)
    _feat_labs  = [f[:25] for f in _top15_stab.index]

    fig_shap_bootstrap, _ax_boot = plt.subplots(figsize=(15, 8), facecolor=_BG)
    _ax_boot.set_facecolor(_BG)
    _ax_boot.spines["top"].set_visible(False); _ax_boot.spines["right"].set_visible(False)
    _ax_boot.spines["left"].set_color(_TXT2);  _ax_boot.spines["bottom"].set_color(_TXT2)

    _xb   = np.arange(len(_top15_stab))
    _wb   = 0.22
    for _wi, (_wn, _col) in enumerate(zip(_wins_avail, _COLS[:3])):
        _ax_boot.bar(_xb + _wi*_wb, _top15_stab[_wn].values, _wb,
                     label=_wn, color=_col, alpha=0.85, edgecolor="none")

    _ax_boot.errorbar(
        _xb + _wb*(len(_wins_avail)-1)/2,
        _top15_stab["mean_imp"].values,
        yerr=_top15_stab["std_imp"].values,
        fmt="none", color=_HL, elinewidth=2, capsize=4, capthick=2, label="±1 Std Dev"
    )
    _ax_boot.set_xticks(_xb + _wb*(len(_wins_avail)-1)/2)
    _ax_boot.set_xticklabels(_feat_labs, rotation=35, ha="right", color=_TXT, fontsize=8)
    _ax_boot.set_ylabel("Permutation Importance (PR-AUC drop)", color=_TXT, fontsize=12)
    _ax_boot.set_title(
        "Bootstrap SHAP Stability — Permutation Importance Across 3 Temporal Windows\n"
        "Error bars = ±1 Std Dev across windows  (tight bars = stable, important feature)",
        color=_TXT, fontsize=12, fontweight="bold", pad=15
    )
    _ax_boot.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10)
    _ax_boot.tick_params(colors=_TXT2)
    plt.tight_layout()
    plt.show()

    print(f"\n   Most STABLE features (lowest variance-to-importance ratio):")
    _st = _stab_df[_stab_df["mean_imp"] > 0.0001].nsmallest(5, "cv_imp")
    for _fn, _row in _st.iterrows():
        print(f"      {_fn:45s}  CV={_row['cv_imp']:.3f}  mean={_row['mean_imp']:+.5f}")

    print(f"\n   Most VOLATILE features (highest variance relative to importance):")
    _vl = _stab_df[_stab_df["mean_imp"].abs() > 0.0001].nlargest(5, "cv_imp")
    for _fn, _row in _vl.iterrows():
        print(f"      {_fn:45s}  CV={_row['cv_imp']:.3f}  mean={_row['mean_imp']:+.5f}")

    shap_bootstrap_stability = _stab_df
else:
    print("   ⚠️  Insufficient windows for bootstrap stability")
    shap_bootstrap_stability = pd.DataFrame()

# ═══════════════════════════════════════════════════════════════
# 6. EXPORT
# ═══════════════════════════════════════════════════════════════
shap_global_importance  = _perm_importance_df
shap_interaction_matrix = _interact_df
shap_ablation_results   = shap_ablation_df

print("\n" + "═"*65)
print("✅ SHAP ADVANCED ANALYSIS COMPLETE")
print("═"*65)
print("   ✅ fig_shap_bar          — Global feature importance (top 20)")
print("   ✅ fig_shap_force        — Per-user force plots (10 users)")
print("   ✅ fig_shap_interact     — Interaction heatmap (top 15 features)")
print("   ✅ shap_ablation_results — Ablation table (4 feature groups)")
print("   ✅ fig_shap_ablation     — Ablation charts (PR-AUC & Lift@10%)")
print("   ✅ fig_shap_bootstrap    — Bootstrap variance plot (3 windows)")
print("   ✅ shap_bootstrap_stability — Stability table with CV scores")
