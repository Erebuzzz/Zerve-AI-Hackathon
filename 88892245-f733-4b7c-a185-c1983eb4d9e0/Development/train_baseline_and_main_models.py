
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════
# 0. DATA PREP — temporal train/val/test splits
# ═════════════════════════════════════════════════════
_TARGETS = ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]
_META_COLS = ["user_id_canon", "split"] + _TARGETS
_FEATURE_COLS = [c for c in modeling_df.columns if c not in _META_COLS]

# Train split has 0 users (train_end=2025-08-01 but data starts 2025-09-01)
# Use val as train and test as test — temporal ordering preserved
_train_mask = modeling_df["split"] == "val"
_test_mask = modeling_df["split"] == "test"

X_train_full = modeling_df.loc[_train_mask, _FEATURE_COLS].values
X_test_full = modeling_df.loc[_test_mask, _FEATURE_COLS].values

# Single feature baseline: feat_active_days
_active_days_idx = _FEATURE_COLS.index("feat_active_days")
X_train_single = X_train_full[:, [_active_days_idx]]
X_test_single = X_test_full[:, [_active_days_idx]]

# Scale features for logistic regression
_scaler_full = StandardScaler()
X_train_scaled = _scaler_full.fit_transform(X_train_full)
X_test_scaled = _scaler_full.transform(X_test_full)

_scaler_single = StandardScaler()
X_train_single_scaled = _scaler_single.fit_transform(X_train_single)
X_test_single_scaled = _scaler_single.transform(X_test_single)

print(f"📋 Split sizes:")
print(f"   Train (val split): {X_train_full.shape[0]} users")
print(f"   Test  (test split): {X_test_full.shape[0]} users")
print(f"   Features: {len(_FEATURE_COLS)}")

# ═════════════════════════════════════════════════════
# 1. HELPER FUNCTIONS
# ═════════════════════════════════════════════════════
def _compute_lift(y_true, y_prob, pct):
    """Compute lift at top pct% of predicted probabilities."""
    _n = len(y_true)
    _k = max(1, int(_n * pct))
    _top_idx = np.argsort(y_prob)[::-1][:_k]
    _top_rate = y_true[_top_idx].mean()
    _base_rate = np.mean(y_true)
    return _top_rate / _base_rate if _base_rate > 0 else 0.0

def _evaluate_model(y_true, y_prob, model_name):
    """Compute all metrics for a model."""
    _y = np.array(y_true)
    _p = np.array(y_prob)
    
    if len(np.unique(_p)) == 1:
        _roc = 0.5
    else:
        _roc = roc_auc_score(_y, _p)
    
    _pr_auc = average_precision_score(_y, _p)
    _lift10 = _compute_lift(_y, _p, 0.10)
    _lift20 = _compute_lift(_y, _p, 0.20)
    
    return {
        "Model": model_name,
        "ROC-AUC": _roc,
        "PR-AUC": _pr_auc,
        "Lift@10%": _lift10,
        "Lift@20%": _lift20,
    }

# ═════════════════════════════════════════════════════
# 2. TRAIN & EVALUATE ALL MODELS FOR EACH TARGET
# ═════════════════════════════════════════════════════
all_results = {}
all_predictions = {}
all_models = {}

for _target in _TARGETS:
    y_train = modeling_df.loc[_train_mask, _target].values
    y_test = modeling_df.loc[_test_mask, _target].values
    
    _pos_rate_train = y_train.mean()
    _pos_rate_test = y_test.mean()
    _n_pos_train = int(y_train.sum())
    _n_neg_train = len(y_train) - _n_pos_train
    
    print(f"\n{'='*65}")
    print(f"🎯 TARGET: {_target}")
    print(f"   Train positive rate: {_pos_rate_train*100:.1f}% ({_n_pos_train}/{len(y_train)})")
    print(f"   Test positive rate:  {_pos_rate_test*100:.1f}% ({int(y_test.sum())}/{len(y_test)})")
    
    _results = []
    
    # --- Baseline 1: Constant predictor (predict base rate) ---
    _const_prob = np.full(len(y_test), _pos_rate_train)
    _res = _evaluate_model(y_test, _const_prob, "Constant (base rate)")
    _results.append(_res)
    all_predictions[(_target, "Constant")] = _const_prob
    
    # --- Baseline 2: Single-feature logistic (feat_active_days) ---
    _lr_single = LogisticRegression(max_iter=1000, random_state=42)
    _lr_single.fit(X_train_single_scaled, y_train)
    _single_prob = _lr_single.predict_proba(X_test_single_scaled)[:, 1]
    _res = _evaluate_model(y_test, _single_prob, "LR (active_days only)")
    _results.append(_res)
    all_predictions[(_target, "LR_single")] = _single_prob
    all_models[(_target, "LR_single")] = _lr_single
    
    # --- Main Model 1: L2 Logistic Regression (all features) ---
    _spw = _n_neg_train / max(_n_pos_train, 1) if _target == "y_upgrade_60d" else 1.0
    _class_weight = {0: 1.0, 1: _spw} if _spw > 1 else None
    
    _lr_full = LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, random_state=42,
        class_weight=_class_weight
    )
    _lr_full.fit(X_train_scaled, y_train)
    _lr_full_prob = _lr_full.predict_proba(X_test_scaled)[:, 1]
    _res = _evaluate_model(y_test, _lr_full_prob, "L2 Logistic Regression")
    _results.append(_res)
    all_predictions[(_target, "L2_LR")] = _lr_full_prob
    all_models[(_target, "L2_LR")] = _lr_full
    
    # --- Main Model 2: HistGradientBoosting (sklearn's LightGBM-like) ---
    # Use sample_weight for class imbalance instead of class_weight param
    _sample_weights = np.ones(len(y_train))
    if _target == "y_upgrade_60d" and _n_pos_train > 0:
        _weight_pos = _n_neg_train / _n_pos_train
        _sample_weights[y_train == 1] = _weight_pos
    
    _hgb_model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=10,
        max_bins=255,
        random_state=42,
        early_stopping=False,
    )
    _hgb_model.fit(X_train_full, y_train, sample_weight=_sample_weights)
    _hgb_prob = _hgb_model.predict_proba(X_test_full)[:, 1]
    _res = _evaluate_model(y_test, _hgb_prob, "Gradient-Boosted Trees")
    _results.append(_res)
    all_predictions[(_target, "GBT")] = _hgb_prob
    all_models[(_target, "GBT")] = _hgb_model
    
    all_results[_target] = _results
    
    _res_df = pd.DataFrame(_results)
    print(f"\n   📊 Test Set Metrics:")
    for _, _row in _res_df.iterrows():
        print(f"   {_row['Model']:30s}  ROC={_row['ROC-AUC']:.4f}  PR-AUC={_row['PR-AUC']:.4f}  "
              f"L@10={_row['Lift@10%']:.2f}  L@20={_row['Lift@20%']:.2f}")

# ═════════════════════════════════════════════════════
# 3. FALLBACK LOGIC — check upgrade positive rate
# ═════════════════════════════════════════════════════
_upgrade_pos_rate_test = modeling_df.loc[_test_mask, "y_upgrade_60d"].mean() * 100
print(f"\n{'='*65}")
print(f"🔄 FALLBACK LOGIC CHECK:")
print(f"   y_upgrade_60d positive rate on test: {_upgrade_pos_rate_test:.1f}%")

if _upgrade_pos_rate_test < 1.0:
    primary_objective = "y_ret_90d"
    fallback_triggered = True
    print(f"   ⚠️  Upgrade rate < 1% → FALLBACK to y_ret_90d as primary objective")
elif _upgrade_pos_rate_test < 0.5:
    primary_objective = "y_ret_90d"
    fallback_triggered = True
    print(f"   ⚠️  Upgrade rate < 0.5% → FALLBACK to y_ret_90d")
else:
    primary_objective = "y_upgrade_60d"
    fallback_triggered = False
    print(f"   ✅ Upgrade rate ≥ 1% → keeping y_upgrade_60d as primary objective")

print(f"   🏆 PRIMARY OBJECTIVE: {primary_objective}")

# ═════════════════════════════════════════════════════
# 4. BUILD DELTA TABLES VS BASELINES
# ═════════════════════════════════════════════════════
model_delta_tables = {}
for _target in _TARGETS:
    _res_df = pd.DataFrame(all_results[_target])
    _baseline_metrics = _res_df[_res_df["Model"] == "Constant (base rate)"].iloc[0]
    
    _delta_rows = []
    for _, _row in _res_df.iterrows():
        _delta_row = {"Model": _row["Model"]}
        for _metric in ["ROC-AUC", "PR-AUC", "Lift@10%", "Lift@20%"]:
            _delta_row[_metric] = _row[_metric]
            _delta_row[f"Δ{_metric}"] = _row[_metric] - _baseline_metrics[_metric]
        _delta_rows.append(_delta_row)
    
    model_delta_tables[_target] = pd.DataFrame(_delta_rows)

print(f"\n{'='*65}")
print(f"📊 FULL COMPARISON TABLES (deltas vs Constant baseline)")
print(f"{'='*65}")

for _target in _TARGETS:
    _is_primary = " ★ PRIMARY" if _target == primary_objective else ""
    print(f"\n🎯 {_target}{_is_primary}")
    _dt = model_delta_tables[_target]
    print(f"{'Model':32s} {'ROC-AUC':>9s} {'ΔROC':>7s} {'PR-AUC':>9s} {'ΔPR':>7s} {'L@10%':>7s} {'ΔL@10':>7s} {'L@20%':>7s} {'ΔL@20':>7s}")
    print("-" * 100)
    for _, _r in _dt.iterrows():
        print(f"{_r['Model']:32s} {_r['ROC-AUC']:9.4f} {_r['ΔROC-AUC']:+7.4f} "
              f"{_r['PR-AUC']:9.4f} {_r['ΔPR-AUC']:+7.4f} "
              f"{_r['Lift@10%']:7.2f} {_r['ΔLift@10%']:+7.2f} "
              f"{_r['Lift@20%']:7.2f} {_r['ΔLift@20%']:+7.2f}")

model_comparison_results = {t: pd.DataFrame(all_results[t]) for t in _TARGETS}

print(f"\n✅ All models trained and evaluated.")
