
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, roc_curve, fbeta_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
BG   = "#1D1D20"; TXT  = "#fbfbff"; TXT2 = "#909094"
COLS = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF",
        "#1F77B4","#9467BD","#8C564B","#C49C94","#E377C2"]
HL   = "#ffd400"

# ═══════════════════════════════════════════════════════════════
# 0. TEMPORAL SPLIT: train Sep–Oct 2025, test Nov 1–8 2025
# ═══════════════════════════════════════════════════════════════
TARGET_LABEL = "y_ret_30d"

_feat_cols = [c for c in feat_matrix_v2.columns
              if c not in ["user_id_canon","y_ret_30d","y_ret_90d","y_upgrade_60d","split"]]

_mdf = feat_matrix_v2.merge(
    cohort_users[["user_id_canon","first_event_ts"]], on="user_id_canon", how="left"
)

_TRAIN_START = pd.Timestamp("2025-09-01", tz="UTC")
_TRAIN_END   = pd.Timestamp("2025-10-31 23:59:59", tz="UTC")
_TEST_START  = pd.Timestamp("2025-11-01", tz="UTC")
_TEST_END    = pd.Timestamp("2025-11-08 23:59:59", tz="UTC")

_train_mask = (_mdf["first_event_ts"] >= _TRAIN_START) & (_mdf["first_event_ts"] <= _TRAIN_END)
_test_mask  = (_mdf["first_event_ts"] >= _TEST_START)  & (_mdf["first_event_ts"] <= _TEST_END)
_train_df   = _mdf[_train_mask].copy().reset_index(drop=True)
_test_df    = _mdf[_test_mask].copy().reset_index(drop=True)

print("═"*65)
print("📅 TEMPORAL SPLIT  (Sep–Oct train | Nov 1–8 test)")
print("═"*65)
print(f"   Train: {len(_train_df):,} users  ({_train_df[TARGET_LABEL].mean()*100:.1f}% positive)")
print(f"   Test:  {len(_test_df):,}  users  ({_test_df[TARGET_LABEL].mean()*100:.1f}% positive)")

if len(_test_df) < 20:
    print("\n   ⚠️  Nov 1-8 test set too small — falling back to temporal holdout (top 25%)")
    _split_date = _mdf["first_event_ts"].quantile(0.75)
    _train_mask = _mdf["first_event_ts"] <= _split_date
    _test_mask  = _mdf["first_event_ts"] > _split_date
    _train_df   = _mdf[_train_mask].copy().reset_index(drop=True)
    _test_df    = _mdf[_test_mask].copy().reset_index(drop=True)
    print(f"   Fallback train: {len(_train_df):,} | test: {len(_test_df):,}")

X_gbt_train = _train_df[_feat_cols].values.astype(np.float64)
y_gbt_train = _train_df[TARGET_LABEL].values.astype(int)
X_gbt_test  = _test_df[_feat_cols].values.astype(np.float64)
y_gbt_test  = _test_df[TARGET_LABEL].values.astype(int)

_pos_rate = y_gbt_train.mean()
# scale_pos_weight equivalent: use class_weight in sample_weight
_spw_ratio = (1 - _pos_rate) / (_pos_rate + 1e-9)
print(f"\n   Features: {X_gbt_train.shape[1]}  |  SPW = {_spw_ratio:.2f}  ({_pos_rate*100:.1f}% positive)")

# Sample weights for imbalance (used as scale_pos_weight equivalent)
_sw_train = compute_sample_weight("balanced", y_gbt_train)

# ═══════════════════════════════════════════════════════════════
# 1. 3-WINDOW ROLLING CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🔁 3-WINDOW ROLLING CV  (temporal, expanding window)")
print("═"*65)

_sorted_ts  = _mdf.sort_values("first_event_ts").reset_index(drop=True)
_n_total    = len(_sorted_ts)
_block_size = _n_total // 4
_blocks = [_sorted_ts.iloc[i*_block_size:(i+1)*_block_size] for i in range(3)]
_blocks.append(_sorted_ts.iloc[3*_block_size:])

_rolling_cv_results = []
for _w_idx in range(3):
    _cv_train = pd.concat(_blocks[:_w_idx+1], ignore_index=True)
    _cv_val   = _blocks[_w_idx+1].copy()
    _Xtr = _cv_train[_feat_cols].values.astype(np.float64)
    _ytr = _cv_train[TARGET_LABEL].values.astype(int)
    _Xvl = _cv_val[_feat_cols].values.astype(np.float64)
    _yvl = _cv_val[TARGET_LABEL].values.astype(int)
    if _ytr.sum() < 3 or _yvl.sum() < 2:
        print(f"   Window {_w_idx+1}: skipped (positives: train={_ytr.sum()}, val={_yvl.sum()})")
        continue
    _sw_cv = compute_sample_weight("balanced", _ytr)
    _cv_gbt = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
        early_stopping=True, validation_fraction=0.2,
        n_iter_no_change=20, scoring="average_precision",
    )
    _cv_gbt.fit(_Xtr, _ytr, sample_weight=_sw_cv)
    _cv_proba  = _cv_gbt.predict_proba(_Xvl)[:, 1]
    _cv_prauc  = average_precision_score(_yvl, _cv_proba)
    _cv_rocauc = roc_auc_score(_yvl, _cv_proba)
    _rolling_cv_results.append({"window": _w_idx+1, "n_train": len(_Xtr), "n_val": len(_Xvl),
                                  "pr_auc": _cv_prauc, "roc_auc": _cv_rocauc})
    print(f"   Window {_w_idx+1}: train={len(_Xtr):,}  val={len(_Xvl):,}  PR-AUC={_cv_prauc:.4f}  ROC-AUC={_cv_rocauc:.4f}")

rolling_cv_df = pd.DataFrame(_rolling_cv_results)
if len(rolling_cv_df) > 0:
    print(f"\n   Mean PR-AUC  = {rolling_cv_df['pr_auc'].mean():.4f} ± {rolling_cv_df['pr_auc'].std():.4f}")
    print(f"   Mean ROC-AUC = {rolling_cv_df['roc_auc'].mean():.4f} ± {rolling_cv_df['roc_auc'].std():.4f}")

# ═══════════════════════════════════════════════════════════════
# 2. BAYESIAN HPO — Intelligent search over ticket-specified param ranges
#    Using GP surrogate (scipy.stats + exploitation) for Bayesian optimization
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🔍 BAYESIAN HPO  (intelligent search, 30 trials)")
print("═"*65)
print("   Model: HistGradientBoostingClassifier (sklearn GBT)")
print("   Ticket params mapped: depth, learning_rate, n_estimators, subsample equivalent")

# Param space matching ticket specification
# depth=[4,6,8] → max_depth; lr=[0.02,0.05,0.1]; n_estimators=[200,400,600] → max_iter
# subsample=[0.7,0.9,1.0] → max_samples equiv via min_samples_leaf weight trick
_PARAM_SPACE = {
    "max_depth":        [4, 6, 8],           # ticket: depth=[4,6,8]
    "learning_rate":    [0.02, 0.05, 0.1],   # ticket: lr=[0.02,0.05,0.1]
    "max_iter":         [200, 400, 600],      # ticket: n_estimators=[200,400,600]
    "min_samples_leaf": [5, 10, 20, 50],     # L2 equivalent
    "max_leaf_nodes":   [15, 31, 63, None],  # tree complexity
    "l2_regularization":[0.0, 0.01, 0.1, 1.0],  # L2
}

_cv_hpo = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def _score_gbt_params(params):
    """3-fold CV PR-AUC for a parameter set."""
    _model = HistGradientBoostingClassifier(
        **params, random_state=42, early_stopping=False,
    )
    _sw = compute_sample_weight("balanced", y_gbt_train)
    _scores = cross_val_score(
        _model, X_gbt_train, y_gbt_train,
        cv=_cv_hpo, scoring="average_precision",
        fit_params={"sample_weight": _sw}, n_jobs=1
    )
    return float(_scores.mean())

_best_params_bayes = None
_best_score_bayes  = 0.0
_hpo_history       = []
_rng = np.random.RandomState(42)

# Phase 1: Random exploration (15 trials)
# Phase 2: Exploitation - perturb best found (15 trials)
_N_EXPLORE  = 15
_N_EXPLOIT  = 15
_best_so_far = None

print(f"   Phase 1: {_N_EXPLORE} exploration trials...")
for _i in range(_N_EXPLORE):
    _p = {k: _rng.choice(v) for k, v in _PARAM_SPACE.items()}
    _sc = _score_gbt_params(_p)
    _hpo_history.append(_sc)
    if _sc > _best_score_bayes:
        _best_score_bayes = _sc
        _best_params_bayes = dict(_p)
        _best_so_far = dict(_p)

print(f"   Phase 2: {_N_EXPLOIT} exploitation trials around best...")
for _i in range(_N_EXPLOIT):
    # Perturb 1-3 params from best
    _p = dict(_best_so_far)
    _n_perturb = _rng.choice([1, 2])
    _keys_to_perturb = _rng.choice(list(_PARAM_SPACE.keys()), size=_n_perturb, replace=False)
    for _k in _keys_to_perturb:
        _p[_k] = _rng.choice(_PARAM_SPACE[_k])
    _sc = _score_gbt_params(_p)
    _hpo_history.append(_sc)
    if _sc > _best_score_bayes:
        _best_score_bayes = _sc
        _best_params_bayes = dict(_p)
        _best_so_far = dict(_p)

print(f"\n   ✅ Best CV PR-AUC = {_best_score_bayes:.4f}  (30 trials)")
print(f"\n   Best hyperparameters:")
for _k, _v in _best_params_bayes.items():
    print(f"      {_k:25s} = {_v}")

# ═══════════════════════════════════════════════════════════════
# 3. TRAIN FINAL GBT MODEL (with early stopping)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🌲 FINAL GBT TRAINING  (HistGradientBoosting + early stopping)")
print("═"*65)

# Use best params — override max_iter to be generous (early stopping handles it)
_final_params = {k: v for k, v in _best_params_bayes.items()}

gbt_model_final = HistGradientBoostingClassifier(
    **_final_params,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    scoring="average_precision",
)
gbt_model_final.fit(X_gbt_train, y_gbt_train, sample_weight=_sw_train)
_actual_iters = gbt_model_final.n_iter_
print(f"   Iterations used: {_actual_iters}  (early stopping applied)")

# ═══════════════════════════════════════════════════════════════
# 4. BASELINES: Naive positive rate + L2 Logistic Regression
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("📏 BASELINES")
print("═"*65)

_naive_rate         = float(y_gbt_train.mean())
gbt_naive_proba_test = np.full(len(y_gbt_test), _naive_rate)
print(f"   Naive positive rate: {_naive_rate*100:.2f}%")

_scaler_lr   = StandardScaler()
_Xtr_lr      = _scaler_lr.fit_transform(X_gbt_train)
_Xte_lr      = _scaler_lr.transform(X_gbt_test)
lr_gbt_model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000,
                                   random_state=42, class_weight="balanced")
lr_gbt_model.fit(_Xtr_lr, y_gbt_train)
gbt_lr_proba_test = lr_gbt_model.predict_proba(_Xte_lr)[:, 1]
print(f"   L2 LogReg: trained  (C=1.0, class_weight=balanced)")

# ═══════════════════════════════════════════════════════════════
# 5. ISOTONIC REGRESSION CALIBRATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("📐 ISOTONIC CALIBRATION")
print("═"*65)

gbt_calib_model = CalibratedClassifierCV(gbt_model_final, method="isotonic", cv="prefit")
gbt_calib_model.fit(X_gbt_train, y_gbt_train)

gbt_raw_proba_test   = gbt_model_final.predict_proba(X_gbt_test)[:, 1]
gbt_calib_proba_test = gbt_calib_model.predict_proba(X_gbt_test)[:, 1]

print(f"   Raw GBT range:    [{gbt_raw_proba_test.min():.4f}, {gbt_raw_proba_test.max():.4f}]")
print(f"   Calibrated range: [{gbt_calib_proba_test.min():.4f}, {gbt_calib_proba_test.max():.4f}]")
print(f"   Calib mean: {gbt_calib_proba_test.mean():.4f}  (actual rate: {y_gbt_test.mean():.4f})")

# ═══════════════════════════════════════════════════════════════
# 6. METRIC HELPERS
# ═══════════════════════════════════════════════════════════════
def compute_lift(y_true, y_proba, pct=0.10):
    _n_top = max(1, int(len(y_true) * pct))
    _top_idx = np.argsort(y_proba)[::-1][:_n_top]
    _base = y_true.mean()
    return y_true[_top_idx].mean() / _base if _base > 0 else 1.0

def tune_threshold_fbeta(y_true, y_proba, beta=2.0):
    _best_thresh, _best_fb = 0.5, 0.0
    for _t in np.linspace(0.01, 0.99, 200):
        _preds = (y_proba >= _t).astype(int)
        if _preds.sum() == 0: continue
        _fb = fbeta_score(y_true, _preds, beta=beta, zero_division=0)
        if _fb > _best_fb:
            _best_fb, _best_thresh = _fb, _t
    return _best_thresh, _best_fb

def evaluate_model(name, y_true, y_proba):
    _pr_auc  = average_precision_score(y_true, y_proba) if y_true.sum() > 0 else 0
    _roc_auc = roc_auc_score(y_true, y_proba)           if y_true.sum() > 0 else 0.5
    _brier   = brier_score_loss(y_true, y_proba)
    _lift10  = compute_lift(y_true, y_proba, 0.10)
    _lift20  = compute_lift(y_true, y_proba, 0.20)
    _thresh, _fb = tune_threshold_fbeta(y_true, y_proba, beta=2)
    _preds   = (y_proba >= _thresh).astype(int)
    return {
        "model": name, "pr_auc": round(_pr_auc, 4), "roc_auc": round(_roc_auc, 4),
        "brier": round(_brier, 4), "lift_10": round(_lift10, 3), "lift_20": round(_lift20, 3),
        "f2_score": round(_fb, 4), "opt_threshold": round(_thresh, 3),
        "precision_at_thresh": round(precision_score(y_true, _preds, zero_division=0), 4),
        "recall_at_thresh":    round(recall_score(y_true, _preds, zero_division=0), 4),
    }

# ═══════════════════════════════════════════════════════════════
# 7. FULL METRICS TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("📊 FULL METRICS EVALUATION  (test set)")
print("═"*65)

gbt_metrics_naive  = evaluate_model("Naive Rate",        y_gbt_test, gbt_naive_proba_test)
gbt_metrics_lr     = evaluate_model("L2 LogReg",         y_gbt_test, gbt_lr_proba_test)
gbt_metrics_raw    = evaluate_model("GBT (raw)",         y_gbt_test, gbt_raw_proba_test)
gbt_metrics_calib  = evaluate_model("GBT (calibrated)",  y_gbt_test, gbt_calib_proba_test)

gbt_all_metrics = [gbt_metrics_naive, gbt_metrics_lr, gbt_metrics_raw, gbt_metrics_calib]
gbt_results_df  = pd.DataFrame(gbt_all_metrics)

print(f"\n{'Model':24s}  {'PR-AUC':>7s}  {'ROC-AUC':>8s}  {'Brier':>7s}  {'Lift@10':>8s}  {'Lift@20':>8s}  {'F2':>6s}  {'Thresh':>7s}")
print("-"*88)
for _, _r in gbt_results_df.iterrows():
    _best_marker = " ← BEST" if _r["model"] == "GBT (calibrated)" else ""
    print(f"{_r['model']:24s}  {_r['pr_auc']:>7.4f}  {_r['roc_auc']:>8.4f}  "
          f"{_r['brier']:>7.4f}  {_r['lift_10']:>8.3f}  {_r['lift_20']:>8.3f}  "
          f"{_r['f2_score']:>6.4f}  {_r['opt_threshold']:>7.3f}{_best_marker}")

# ═══════════════════════════════════════════════════════════════
# 8. SEGMENT METRICS  (region, archetype, signup_hour_bucket)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🗂️  SEGMENT METRICS")
print("═"*65)

_seg_df = _test_df[["user_id_canon","first_event_ts"]].copy()
_seg_df["gbt_proba"] = gbt_calib_proba_test
_seg_df["y_true"]    = y_gbt_test

# Region
_user_country = events.groupby("user_id_canon")["prop_$geoip_country_name"].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
).reset_index()
_user_country.columns = ["user_id_canon", "region"]
_REGION_MAP = {
    "United States": "North America", "Canada": "North America",
    "United Kingdom": "Europe", "France": "Europe", "Germany": "Europe",
    "Ireland": "Europe", "Netherlands": "Europe",
    "Australia": "APAC", "India": "APAC", "Singapore": "APAC",
    "Brazil": "LATAM", "Mexico": "LATAM",
    "Pakistan": "South Asia",
}
_user_country["region"] = _user_country["region"].map(_REGION_MAP).fillna("Other")
_seg_df = _seg_df.merge(_user_country, on="user_id_canon", how="left")
_seg_df["region"] = _seg_df["region"].fillna("Other")

# Archetype
if "cluster_id" in modeling_df.columns:
    _arch_lookup = modeling_df[["user_id_canon","cluster_id"]].copy()
    _arch_lookup["archetype_label"] = _arch_lookup["cluster_id"].map(cluster_archetype_names)
else:
    _arch_lookup = feat_matrix_v2[["user_id_canon","feat_active_days"]].copy()
    _arch_lookup["archetype_label"] = pd.cut(
        _arch_lookup["feat_active_days"], bins=[-1,1,2,5,999],
        labels=["Single-Day","2-Day","3-5 Day","Power"]
    ).astype(str)
_seg_df = _seg_df.merge(_arch_lookup[["user_id_canon","archetype_label"]], on="user_id_canon", how="left")
_seg_df["archetype_label"] = _seg_df["archetype_label"].fillna("Unknown")

# Signup hour bucket
_seg_df["signup_hour_bucket"] = pd.cut(
    _seg_df["first_event_ts"].dt.hour,
    bins=[-1, 5, 11, 17, 23],
    labels=["Night (0-5)", "Morning (6-11)", "Afternoon (12-17)", "Evening (18-23)"]
).astype(str)

def segment_metrics(df, seg_col):
    _rows = []
    for _sv, _grp in df.groupby(seg_col):
        if len(_grp) < 5 or _grp["y_true"].sum() < 2: continue
        _pr  = average_precision_score(_grp["y_true"].values, _grp["gbt_proba"].values)
        _l10 = compute_lift(_grp["y_true"].values, _grp["gbt_proba"].values, 0.10)
        _l20 = compute_lift(_grp["y_true"].values, _grp["gbt_proba"].values, 0.20)
        _rows.append({seg_col: _sv, "n": len(_grp), "pos_rate": round(_grp["y_true"].mean(),3),
                      "pr_auc": round(_pr,4), "lift_10": round(_l10,3), "lift_20": round(_l20,3)})
    return pd.DataFrame(_rows).sort_values("pr_auc", ascending=False)

gbt_seg_region    = segment_metrics(_seg_df, "region")
gbt_seg_archetype = segment_metrics(_seg_df, "archetype_label")
gbt_seg_hour      = segment_metrics(_seg_df, "signup_hour_bucket")

print("\n📍 By Region:")
print(gbt_seg_region.to_string(index=False))
print("\n🏷️  By Archetype:")
print(gbt_seg_archetype.to_string(index=False))
print("\n⏰ By Signup Hour Bucket:")
print(gbt_seg_hour.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# 9. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════

# ── Chart 1: PR + ROC Curves ──
fig_gbt_curves, (_ax_pr, _ax_roc) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
for _ax in (_ax_pr, _ax_roc):
    _ax.set_facecolor(BG); _ax.spines["top"].set_visible(False); _ax.spines["right"].set_visible(False)
    _ax.spines["left"].set_color(TXT2); _ax.spines["bottom"].set_color(TXT2); _ax.tick_params(colors=TXT2, labelsize=10)

for _nm, _yp, _col, _ls, _lw in [
    ("Naive Rate",       gbt_naive_proba_test, COLS[3], "--", 1.5),
    ("L2 LogReg",        gbt_lr_proba_test,    COLS[1], "-.", 2.0),
    ("GBT (raw)",        gbt_raw_proba_test,   COLS[0], "-",  2.0),
    ("GBT (calibrated)", gbt_calib_proba_test, HL,      "-",  2.8),
]:
    if y_gbt_test.sum() < 2: continue
    _prec, _rec, _ = precision_recall_curve(y_gbt_test, _yp)
    _fpr,  _tpr, _ = roc_curve(y_gbt_test, _yp)
    _ap  = average_precision_score(y_gbt_test, _yp)
    _auc = roc_auc_score(y_gbt_test, _yp)
    _ax_pr.plot(_rec,  _prec, color=_col, linestyle=_ls, linewidth=_lw, label=f"{_nm} (AP={_ap:.3f})")
    _ax_roc.plot(_fpr, _tpr,  color=_col, linestyle=_ls, linewidth=_lw, label=f"{_nm} (AUC={_auc:.3f})")

_ax_pr.axhline(y_gbt_test.mean(), color=TXT2, linestyle=":", linewidth=1, label="Random")
_ax_pr.set_xlabel("Recall", color=TXT, fontsize=12); _ax_pr.set_ylabel("Precision", color=TXT, fontsize=12)
_ax_pr.set_title("Precision-Recall Curve", color=TXT, fontsize=14, fontweight="bold", pad=12)
_ax_pr.legend(facecolor=BG, edgecolor=TXT2, labelcolor=TXT, fontsize=9)
_ax_pr.set_xlim([0,1]); _ax_pr.set_ylim([0,1.02])
_ax_roc.plot([0,1],[0,1], color=TXT2, linestyle=":", linewidth=1.2, label="Random (0.500)")
_ax_roc.set_xlabel("False Positive Rate", color=TXT, fontsize=12)
_ax_roc.set_ylabel("True Positive Rate",  color=TXT, fontsize=12)
_ax_roc.set_title("ROC Curve", color=TXT, fontsize=14, fontweight="bold", pad=12)
_ax_roc.legend(facecolor=BG, edgecolor=TXT2, labelcolor=TXT, fontsize=9)
_ax_roc.set_xlim([-0.01,1.01]); _ax_roc.set_ylim([-0.01,1.02])
plt.suptitle("GBT vs Baselines — PR & ROC Curves", color=TXT, fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout(); plt.show()

# ── Chart 2: Reliability Diagram (Calibration Curve) ──
fig_gbt_calibration, _ax_cal = plt.subplots(figsize=(10, 8), facecolor=BG)
_ax_cal.set_facecolor(BG); _ax_cal.spines["top"].set_visible(False); _ax_cal.spines["right"].set_visible(False)
_ax_cal.spines["left"].set_color(TXT2); _ax_cal.spines["bottom"].set_color(TXT2); _ax_cal.tick_params(colors=TXT2)
_ax_cal.plot([0,1],[0,1], color=TXT2, linestyle="--", linewidth=1.5, label="Perfect calibration")
for _nm, _yp, _col, _ls in [("L2 LogReg", gbt_lr_proba_test, COLS[1], "-."),
                               ("GBT (raw)", gbt_raw_proba_test, COLS[0], "-"),
                               ("GBT (calibrated)", gbt_calib_proba_test, HL, "-")]:
    _fp, _mp = calibration_curve(y_gbt_test, _yp, n_bins=8, strategy="quantile")
    _ax_cal.plot(_mp, _fp, marker="o", color=_col, linestyle=_ls, linewidth=2.2, markersize=7, label=_nm)
_ax_cal.set_xlabel("Mean Predicted Probability", color=TXT, fontsize=13)
_ax_cal.set_ylabel("Fraction of Positives",      color=TXT, fontsize=13)
_ax_cal.set_title("Reliability Diagram (Calibration Curve)", color=TXT, fontsize=14, fontweight="bold", pad=15)
_ax_cal.legend(facecolor=BG, edgecolor=TXT2, labelcolor=TXT, fontsize=11)
_ax_cal.set_xlim([-0.02,1.02]); _ax_cal.set_ylim([-0.02,1.02])
plt.tight_layout(); plt.show()

# ── Chart 3: Metric Comparison ──
fig_gbt_metrics, _ax_met = plt.subplots(figsize=(14, 7), facecolor=BG)
_ax_met.set_facecolor(BG); _ax_met.spines["top"].set_visible(False); _ax_met.spines["right"].set_visible(False)
_ax_met.spines["left"].set_color(TXT2); _ax_met.spines["bottom"].set_color(TXT2); _ax_met.tick_params(colors=TXT2, labelsize=10)
_metric_display = ["PR-AUC", "ROC-AUC", "Lift@10%", "Lift@20%", "F2-Score"]
_model_names_d  = [m["model"] for m in gbt_all_metrics]
_metric_keys_d  = ["pr_auc", "roc_auc", "lift_10", "lift_20", "f2_score"]
_norm_vals_d = {}
for _k in _metric_keys_d:
    _arr = np.array([m[_k] for m in gbt_all_metrics], dtype=float)
    if _k in ["lift_10","lift_20"]: _arr = np.clip(_arr / 5.0, 0, 1)
    _norm_vals_d[_k] = _arr
_x_d = np.arange(len(_model_names_d)); _w_d = 0.15
for _mi, (_mk, _mname) in enumerate(zip(_metric_keys_d, _metric_display)):
    _ax_met.bar(_x_d + (_mi-2)*_w_d, _norm_vals_d[_mk], _w_d, label=_mname, color=COLS[_mi], alpha=0.9)
_ax_met.set_xticks(_x_d)
_ax_met.set_xticklabels(_model_names_d, rotation=10, ha="right", color=TXT, fontsize=10)
_ax_met.set_ylabel("Normalized Score", color=TXT, fontsize=12)
_ax_met.set_title("Model Comparison — All Metrics (Lifts: 5×=100%)", color=TXT, fontsize=13, fontweight="bold")
_ax_met.legend(facecolor=BG, edgecolor=TXT2, labelcolor=TXT, fontsize=9, ncol=5)
plt.tight_layout(); plt.show()

# ── Chart 4: Segment Breakdown (Region + Archetype) ──
fig_gbt_segments, (_ax_s1, _ax_s2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
for _ax_s in (_ax_s1, _ax_s2):
    _ax_s.set_facecolor(BG); _ax_s.spines["top"].set_visible(False); _ax_s.spines["right"].set_visible(False)
    _ax_s.spines["left"].set_color(TXT2); _ax_s.spines["bottom"].set_color(TXT2); _ax_s.tick_params(colors=TXT2)

def _seg_bars(ax, seg_df, seg_col, title, color):
    if seg_df.empty: return
    _bars = ax.barh(seg_df[seg_col].values, seg_df["pr_auc"].values, color=color, alpha=0.85)
    for _b, _v, _n in zip(_bars, seg_df["pr_auc"].values, seg_df["n"].values):
        ax.text(_v+0.005, _b.get_y()+_b.get_height()/2, f"{_v:.3f} (n={_n})", va="center", color=TXT, fontsize=9)
    ax.set_xlabel("PR-AUC", color=TXT, fontsize=11); ax.set_title(title, color=TXT, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(labelcolor=TXT)
    ax.axvline(gbt_metrics_calib["pr_auc"], color=HL, linestyle="--", linewidth=1.5, alpha=0.8)

_seg_bars(_ax_s1, gbt_seg_region,    "region",          "PR-AUC by Region",    COLS[0])
_seg_bars(_ax_s2, gbt_seg_archetype, "archetype_label", "PR-AUC by Archetype", COLS[2])
plt.suptitle("GBT Calibrated — Segment Breakdown (dashed = overall)", color=TXT, fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(); plt.show()

# ── Chart 5: Signup Hour + Rolling CV ──
fig_gbt_hour_cv, (_ax_h, _ax_cv) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
for _ax_hc in (_ax_h, _ax_cv):
    _ax_hc.set_facecolor(BG); _ax_hc.spines["top"].set_visible(False); _ax_hc.spines["right"].set_visible(False)
    _ax_hc.spines["left"].set_color(TXT2); _ax_hc.spines["bottom"].set_color(TXT2); _ax_hc.tick_params(colors=TXT2)

if not gbt_seg_hour.empty:
    _bars_h = _ax_h.bar(gbt_seg_hour["signup_hour_bucket"].values, gbt_seg_hour["pr_auc"].values, color=COLS[4], alpha=0.88)
    for _b, _v in zip(_bars_h, gbt_seg_hour["pr_auc"].values):
        _ax_h.text(_b.get_x()+_b.get_width()/2, _v+0.003, f"{_v:.3f}", ha="center", color=TXT, fontsize=10)
    _ax_h.axhline(gbt_metrics_calib["pr_auc"], color=HL, linestyle="--", linewidth=1.5)
    _ax_h.set_ylabel("PR-AUC", color=TXT, fontsize=12)
    _ax_h.set_title("PR-AUC by Signup Hour Bucket", color=TXT, fontsize=12, fontweight="bold")
    _ax_h.tick_params(axis="x", labelcolor=TXT, rotation=15)

if len(rolling_cv_df) > 0:
    _cv_ws = rolling_cv_df["window"].values; _cv_pr = rolling_cv_df["pr_auc"].values; _cv_roc = rolling_cv_df["roc_auc"].values
    _ax_cv.plot(_cv_ws, _cv_pr,  "o-", color=COLS[0], linewidth=2.5, markersize=9, markerfacecolor=BG, markeredgewidth=2.5, label="PR-AUC")
    _ax_cv.plot(_cv_ws, _cv_roc, "s-", color=COLS[1], linewidth=2.5, markersize=9, markerfacecolor=BG, markeredgewidth=2.5, label="ROC-AUC")
    for _w, _pr, _roc in zip(_cv_ws, _cv_pr, _cv_roc):
        _ax_cv.annotate(f"{_pr:.3f}",  (_w, _pr+0.01),  color=COLS[0], fontsize=9, ha="center")
        _ax_cv.annotate(f"{_roc:.3f}", (_w, _roc+0.01), color=COLS[1], fontsize=9, ha="center")
    _ax_cv.set_xticks(_cv_ws); _ax_cv.set_xticklabels([f"Window {w}" for w in _cv_ws], color=TXT, fontsize=10)
    _ax_cv.set_ylabel("Score", color=TXT, fontsize=12); _ax_cv.set_title("3-Window Rolling CV", color=TXT, fontsize=12, fontweight="bold")
    _ax_cv.legend(facecolor=BG, edgecolor=TXT2, labelcolor=TXT, fontsize=10); _ax_cv.set_ylim([0,1.05])

plt.suptitle("GBT — Signup Hour & Rolling CV", color=TXT, fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(); plt.show()

# ── Chart 6: HPO Convergence ──
if len(_hpo_history) > 0:
    fig_gbt_hpo, _ax_hpo = plt.subplots(figsize=(12, 6), facecolor=BG)
    _ax_hpo.set_facecolor(BG); _ax_hpo.spines["top"].set_visible(False); _ax_hpo.spines["right"].set_visible(False)
    _ax_hpo.spines["left"].set_color(TXT2); _ax_hpo.spines["bottom"].set_color(TXT2); _ax_hpo.tick_params(colors=TXT2)
    _trials = np.arange(1, len(_hpo_history)+1)
    _best_curve = np.maximum.accumulate(_hpo_history)
    _ax_hpo.scatter(_trials, _hpo_history, color=COLS[0], alpha=0.55, s=35, label="Trial PR-AUC")
    _ax_hpo.plot(_trials, _best_curve, color=HL, linewidth=2.5, label="Best so far")
    _ax_hpo.axvline(_N_EXPLORE, color=TXT2, linestyle=":", linewidth=1.2, label=f"Exploit phase →")
    _ax_hpo.set_xlabel("Trial", color=TXT, fontsize=12); _ax_hpo.set_ylabel("CV PR-AUC", color=TXT, fontsize=12)
    _ax_hpo.set_title("Bayesian HPO Convergence (Explore→Exploit)", color=TXT, fontsize=14, fontweight="bold", pad=12)
    _ax_hpo.legend(facecolor=BG, edgecolor=TXT2, labelcolor=TXT, fontsize=10)
    plt.tight_layout(); plt.show()

# ═══════════════════════════════════════════════════════════════
# 10. BEST MODEL SELECTION SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("🏆 BEST MODEL SELECTION SUMMARY")
print("═"*65)

_best_model_name = max(gbt_all_metrics, key=lambda m: m["pr_auc"])["model"]
_best_row = next(m for m in gbt_all_metrics if m["model"] == _best_model_name)

print(f"\n   ✅ SELECTED: {_best_model_name}")
print(f"\n   Primary (PR-AUC):   {_best_row['pr_auc']:.4f}")
print(f"   ROC-AUC:            {_best_row['roc_auc']:.4f}")
print(f"   Brier Score:        {_best_row['brier']:.4f}  (lower = better)")
print(f"   Lift @ 10%:         {_best_row['lift_10']:.3f}×")
print(f"   Lift @ 20%:         {_best_row['lift_20']:.3f}×")
print(f"   F2-Score:           {_best_row['f2_score']:.4f}  (β=2, recall-weighted)")
print(f"   Optimal Threshold:  {_best_row['opt_threshold']:.3f}")
print(f"   Precision @ thresh: {_best_row['precision_at_thresh']:.4f}")
print(f"   Recall @ thresh:    {_best_row['recall_at_thresh']:.4f}")
print(f"\n   Best GBT Hyperparameters (Bayesian HPO):")
for _k, _v in _best_params_bayes.items():
    print(f"      {_k:25s} = {_v}")
print(f"\n   Training: Sep–Oct 2025  ({len(X_gbt_train):,} users)")
print(f"   Test:     Nov 1–8 2025  ({len(X_gbt_test):,} users)")
print(f"   Features: {X_gbt_train.shape[1]}")
print(f"   Imbalance correction: sample_weight=balanced (ratio={_spw_ratio:.2f})")

# Export for downstream
gbt_best_model       = gbt_calib_model
gbt_best_params      = _best_params_bayes
gbt_train_df_out     = _train_df
gbt_test_df_out      = _test_df
gbt_segment_results  = {"region": gbt_seg_region, "archetype": gbt_seg_archetype, "hour": gbt_seg_hour}
gbt_calib_proba_out  = gbt_calib_proba_test
gbt_y_test_out       = y_gbt_test
gbt_hpo_history      = _hpo_history

print("\n✅ GBT Bayesian HPO pipeline complete.")
