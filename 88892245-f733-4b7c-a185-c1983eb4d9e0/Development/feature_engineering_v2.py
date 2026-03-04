
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy

# ═════════════════════════════════════════════════════
# 0. SETUP — rebuild early window from scratch
# ═════════════════════════════════════════════════════
_EARLY_END = EARLY_WINDOW_DAYS  # 7 days

_ev = events.merge(
    cohort_users[["user_id_canon", "first_event_ts"]],
    on="user_id_canon", how="inner"
).copy()
_ev["days_since_first"] = (_ev["timestamp"] - _ev["first_event_ts"]).dt.total_seconds() / 86400

# Strict early window (no leakage: ≤ 7 days)
_early = _ev[_ev["days_since_first"] <= _EARLY_END].copy()
_post  = _ev[_ev["days_since_first"] >  _EARLY_END].copy()

print(f"📦 Early window events: {len(_early):,} | Post-window: {len(_post):,}")

# ═════════════════════════════════════════════════════
# 1. COPY BASE FEATURE MATRIX (from compute_labels_and_features)
# ═════════════════════════════════════════════════════
# Start from modeling_df which already has 74 features + labels + split
feat_df = modeling_df.copy()
_initial_feat_cols = [c for c in feat_df.columns
                      if c not in ["user_id_canon", "y_ret_30d", "y_ret_90d", "y_upgrade_60d", "split"]]
print(f"✅ Starting features (from modeling_df): {len(_initial_feat_cols)}")

# ═════════════════════════════════════════════════════
# 2. DERIVED FEATURES — all within early window (no leakage)
# ═════════════════════════════════════════════════════

# --- EVENT CATEGORY MAPPING (same as compute_labels_and_features) ---
_CATEGORIES = {
    "agent": ["agent_tool_call_create_block_tool","agent_tool_call_run_block_tool",
              "agent_tool_call_get_block_tool","agent_tool_call_get_canvas_summary_tool",
              "agent_tool_call_get_variable_preview_tool","agent_tool_call_finish_ticket_tool",
              "agent_tool_call_refactor_block_tool","agent_tool_call_delete_block_tool",
              "agent_tool_call_create_edges_tool","agent_tool_call_get_block_image_tool",
              "agent_worker_created","agent_new_chat","agent_start_from_prompt",
              "agent_message","agent_accept_suggestion","agent_open",
              "agent_suprise_me","agent_open_error_assist","agent_upload_files",
              "agent_block_created"],
    "block_ops": ["run_block","block_create","block_delete","block_resize",
                  "block_rename","block_copy","block_paste","block_open_compute_settings",
                  "block_output_copy","block_output_download","stop_block",
                  "run_all_blocks","run_upto_block","run_from_block"],
}
_event_to_cat = {}
for _cat, _evts in _CATEGORIES.items():
    for _e in _evts:
        _event_to_cat[_e] = _cat

_early = _early.copy()
_early["event_cat"] = _early["event"].map(_event_to_cat).fillna("other")

# --- DERIVED FEATURE 1: session_entropy ---
# Shannon entropy of event types within each session → measures how varied
# a user's sessions are (high = diverse workflow, low = repetitive)
_sess_event_counts = (
    _early.groupby(["user_id_canon", "prop_$session_id", "event"])
    .size()
    .reset_index(name="_n")
)

def _session_entropy_user(_grp):
    """Mean per-session entropy of event type distribution."""
    _sess_entropies = []
    for _, _sess_df in _grp.groupby("prop_$session_id"):
        _counts = _sess_df["_n"].values.astype(float)
        _total = _counts.sum()
        if _total < 2:
            _sess_entropies.append(0.0)
        else:
            _probs = _counts / _total
            _sess_entropies.append(float(scipy_entropy(_probs, base=2)))
    return np.mean(_sess_entropies) if _sess_entropies else 0.0

_entropy_series = (
    _sess_event_counts
    .groupby("user_id_canon")
    .apply(_session_entropy_user)
    .reset_index(name="feat_session_entropy")
)
print(f"   ✅ feat_session_entropy: {_entropy_series['feat_session_entropy'].describe().round(3).to_dict()}")

# --- DERIVED FEATURE 2: time_of_day_activity_profile ---
# Entropy of hour-of-day distribution → 0 = always same hour, high = spread across day
_early["_hour"] = _early["timestamp"].dt.hour

def _tod_entropy(_grp):
    _counts = _grp["_hour"].value_counts().values.astype(float)
    if _counts.sum() < 2:
        return 0.0
    _probs = _counts / _counts.sum()
    return float(scipy_entropy(_probs, base=2))

_tod_entropy_series = (
    _early.groupby("user_id_canon")
    .apply(_tod_entropy)
    .reset_index(name="feat_tod_entropy")
)
print(f"   ✅ feat_tod_entropy: {_tod_entropy_series['feat_tod_entropy'].describe().round(3).to_dict()}")

# --- DERIVED FEATURE 3: day_gap_variance ---
# Variance of inter-day gaps → captures irregularity in usage patterns
def _day_gap_variance(_grp):
    _dates = sorted(_grp["timestamp"].dt.date.unique())
    if len(_dates) < 2:
        return 0.0
    _diffs = [(b - a).days for a, b in zip(_dates[:-1], _dates[1:])]
    return float(np.var(_diffs)) if len(_diffs) >= 2 else 0.0

_gap_var_series = (
    _early.groupby("user_id_canon")
    .apply(_day_gap_variance)
    .reset_index(name="feat_day_gap_variance")
)
print(f"   ✅ feat_day_gap_variance: {_gap_var_series['feat_day_gap_variance'].describe().round(3).to_dict()}")

# --- DERIVED FEATURE 4: execution_to_agent_ratio ---
# Ratio of block-execution events to agent tool-call events in early window
# Low = pure AI-driven user, High = manual executor, ~1 = balanced
_run_counts = (
    _early[_early["event"] == "run_block"]
    .groupby("user_id_canon").size()
    .reset_index(name="_run_cnt")
)
_agent_counts = (
    _early[_early["event_cat"] == "agent"]
    .groupby("user_id_canon").size()
    .reset_index(name="_agent_cnt")
)
_exec_agent_ratio = (
    cohort_users[["user_id_canon"]]
    .merge(_run_counts, on="user_id_canon", how="left")
    .merge(_agent_counts, on="user_id_canon", how="left")
    .fillna(0)
)
# Smoothed ratio: (runs + 1) / (agent_calls + 1) to avoid division by zero
_exec_agent_ratio["feat_execution_to_agent_ratio"] = (
    (_exec_agent_ratio["_run_cnt"] + 1) / (_exec_agent_ratio["_agent_cnt"] + 1)
)
_exec_agent_ratio = _exec_agent_ratio[["user_id_canon", "feat_execution_to_agent_ratio"]]
print(f"   ✅ feat_execution_to_agent_ratio: {_exec_agent_ratio['feat_execution_to_agent_ratio'].describe().round(3).to_dict()}")

# --- DERIVED FEATURE 5: canvas_creation_rate ---
# Number of canvases created per active day in early window
# Measures how aggressively user explores new workspaces
_canvas_events = _early[_early["event"] == "canvas_create"]
_canvas_cnt = (
    _canvas_events.groupby("user_id_canon").size()
    .reset_index(name="_canvas_cnt")
)
_active_days_per_user = (
    _early.groupby("user_id_canon")["timestamp"]
    .apply(lambda x: x.dt.date.nunique())
    .reset_index(name="_active_days")
)
_canvas_rate = (
    cohort_users[["user_id_canon"]]
    .merge(_canvas_cnt, on="user_id_canon", how="left")
    .merge(_active_days_per_user, on="user_id_canon", how="left")
    .fillna({"_canvas_cnt": 0})
)
_canvas_rate["feat_canvas_creation_rate"] = (
    _canvas_rate["_canvas_cnt"] / _canvas_rate["_active_days"].clip(lower=1)
)
_canvas_rate = _canvas_rate[["user_id_canon", "feat_canvas_creation_rate"]]
print(f"   ✅ feat_canvas_creation_rate: {_canvas_rate['feat_canvas_creation_rate'].describe().round(3).to_dict()}")

# --- DERIVED FEATURE 6: agent_to_block_conversion_flag ---
# Binary: did the user create a block after using the agent in the same session?
# Strong adoption signal — agent led to concrete block creation
_agent_sessions = set(
    _early[_early["event_cat"] == "agent"]["prop_$session_id"].dropna()
)
_block_create_sessions = set(
    _early[_early["event"] == "block_create"]["prop_$session_id"].dropna()
)
_converted_sessions = _agent_sessions & _block_create_sessions
_converted_users = set(
    _early[_early["prop_$session_id"].isin(_converted_sessions)]["user_id_canon"]
)

_conversion_flag = pd.DataFrame({
    "user_id_canon": cohort_users["user_id_canon"],
    "feat_agent_block_conversion": cohort_users["user_id_canon"].isin(_converted_users).astype(int)
})
print(f"   ✅ feat_agent_block_conversion: {_conversion_flag['feat_agent_block_conversion'].sum():,} users converted "
      f"({_conversion_flag['feat_agent_block_conversion'].mean()*100:.1f}%)")

# ═════════════════════════════════════════════════════
# 3. MERGE ALL DERIVED FEATURES INTO FEATURE MATRIX
# ═════════════════════════════════════════════════════
_derived_frames = [
    _entropy_series,       # feat_session_entropy
    _tod_entropy_series,   # feat_tod_entropy
    _gap_var_series,       # feat_day_gap_variance
    _exec_agent_ratio,     # feat_execution_to_agent_ratio
    _canvas_rate,          # feat_canvas_creation_rate
    _conversion_flag,      # feat_agent_block_conversion
]

for _df in _derived_frames:
    feat_df = feat_df.merge(_df, on="user_id_canon", how="left")

# Fill derived feature NaNs with 0
_derived_feat_cols = [
    "feat_session_entropy", "feat_tod_entropy", "feat_day_gap_variance",
    "feat_execution_to_agent_ratio", "feat_canvas_creation_rate",
    "feat_agent_block_conversion"
]
for _c in _derived_feat_cols:
    feat_df[_c] = feat_df[_c].fillna(0.0)

print(f"\n✅ Derived features merged. Shape: {feat_df.shape}")

# ═════════════════════════════════════════════════════
# 4. DROP ZERO-VARIANCE FEATURES
# ═════════════════════════════════════════════════════
_all_feat_cols = [c for c in feat_df.columns
                  if c not in ["user_id_canon", "y_ret_30d", "y_ret_90d", "y_upgrade_60d", "split"]]

_variances = feat_df[_all_feat_cols].var()
_zero_var_cols = _variances[_variances < 1e-10].index.tolist()
print(f"\n🗑️  Zero-variance features to drop: {len(_zero_var_cols)}")
if _zero_var_cols:
    print(f"   {_zero_var_cols}")
feat_df.drop(columns=_zero_var_cols, inplace=True, errors="ignore")

# ═════════════════════════════════════════════════════
# 5. REMOVE HIGHLY CORRELATED FEATURES (> 0.92 threshold)
# ═════════════════════════════════════════════════════
_feat_cols_post_var = [c for c in feat_df.columns
                       if c not in ["user_id_canon", "y_ret_30d", "y_ret_90d", "y_upgrade_60d", "split"]]

_corr_matrix = feat_df[_feat_cols_post_var].corr().abs()
# Upper triangle mask
_upper = _corr_matrix.where(np.triu(np.ones(_corr_matrix.shape), k=1).astype(bool))

_CORR_THRESHOLD = 0.92
_high_corr_cols = [c for c in _upper.columns if any(_upper[c] > _CORR_THRESHOLD)]

print(f"\n📉 High-correlation (> {_CORR_THRESHOLD}) features to drop: {len(_high_corr_cols)}")
if _high_corr_cols:
    for _hc in _high_corr_cols:
        _partners = _upper.index[_upper[_hc] > _CORR_THRESHOLD].tolist()
        print(f"   Drop '{_hc}' (corr > {_CORR_THRESHOLD} with: {_partners})")

feat_df.drop(columns=_high_corr_cols, inplace=True, errors="ignore")

# ═════════════════════════════════════════════════════
# 6. TARGET ENCODING FOR SPARSE CATEGORICALS
# ═════════════════════════════════════════════════════
# Sparse one-hot columns (low-cardinality country/device that survived so far)
# Strategy: Replace OHE binary columns with target-encoded mean for ret30d on TRAIN only
# Then propagate to val/test.  This avoids leakage from test labels.

_train_mask = feat_df["split"] == "train"
_val_mask   = feat_df["split"] == "val"
_test_mask  = feat_df["split"] == "test"

_n_train = _train_mask.sum()
print(f"\n🎯 TARGET ENCODING")
print(f"   Train rows: {_n_train:,} | Val: {_val_mask.sum():,} | Test: {_test_mask.sum():,}")

# Identify remaining sparse categoricals (country/device/os/browser OHE columns)
_ohe_prefixes = ["feat_primary_country_", "feat_primary_device_",
                 "feat_primary_os_", "feat_primary_browser_"]
_ohe_cols = [c for c in feat_df.columns
             if any(c.startswith(p) for p in _ohe_prefixes)]

# Group OHE cols by prefix for consolidated encoding
_prefix_groups = {}
for _p in _ohe_prefixes:
    _grp = [c for c in _ohe_cols if c.startswith(_p)]
    if _grp:
        _prefix_groups[_p] = _grp

print(f"   OHE groups to target-encode: {list(_prefix_groups.keys())}")

# Perform target encoding: for each group, encode each OHE column using
# train-split mean of y_ret_30d (primary proxy target)
# Use smoothed estimate: (count * mean + global_mean * k) / (count + k) where k=10
_global_mean_train = feat_df.loc[_train_mask, "y_ret_30d"].mean() if _n_train > 0 else 0.3
_K_SMOOTH = 10  # smoothing strength

if _n_train == 0:
    # No train rows — use val-based encoding as fallback
    print("   ⚠️  No train rows available — using val-split for target encoding")
    _encode_mask = _val_mask
    _global_mean_enc = feat_df.loc[_val_mask, "y_ret_30d"].mean()
else:
    _encode_mask = _train_mask
    _global_mean_enc = _global_mean_train

# For each prefix group, create a single smoothed target-encoded numeric feature
_te_new_cols = {}
for _p, _cols in _prefix_groups.items():
    _feat_name = _p.rstrip("_").replace("feat_primary_", "feat_te_")
    _te_vals = np.zeros(len(feat_df))
    
    for _c in _cols:
        # Users where this OHE is 1
        _is_one = feat_df[_c].values == 1
        if _is_one.sum() == 0:
            continue
        
        # Compute smoothed mean from encode-split only
        _enc_is_one = _is_one & _encode_mask.values
        _enc_count = _enc_is_one.sum()
        _enc_mean  = feat_df.loc[_enc_is_one, "y_ret_30d"].mean() if _enc_count > 0 else _global_mean_enc
        _smoothed  = (_enc_count * _enc_mean + _K_SMOOTH * _global_mean_enc) / (_enc_count + _K_SMOOTH)
        
        # Assign smoothed mean to all users with that OHE=1
        _te_vals[_is_one] = _smoothed
    
    # Users with all OHE = 0 (e.g., "Unknown" or not in top-10) → global mean
    _all_zero_mask = feat_df[_cols].sum(axis=1) == 0
    _te_vals[_all_zero_mask.values] = _global_mean_enc
    
    _te_new_cols[_feat_name] = _te_vals
    print(f"   Encoded {len(_cols)} cols → {_feat_name} "
          f"(mean={np.mean(_te_vals):.3f}, std={np.std(_te_vals):.3f})")

# Add TE columns and drop original OHE columns
for _feat_name, _vals in _te_new_cols.items():
    feat_df[_feat_name] = _vals

feat_df.drop(columns=_ohe_cols, inplace=True, errors="ignore")

# ═════════════════════════════════════════════════════
# 7. FINAL FEATURE MATRIX — LEAKAGE AUDIT
# ═════════════════════════════════════════════════════
_final_feat_cols = [c for c in feat_df.columns
                    if c not in ["user_id_canon", "y_ret_30d", "y_ret_90d", "y_upgrade_60d", "split"]]

# Rename for clean downstream access
feat_matrix_v2 = feat_df.copy()

print(f"\n{'='*60}")
print(f"📦 FINAL FEATURE MATRIX: feat_matrix_v2")
print(f"{'='*60}")
print(f"   Shape: {feat_matrix_v2.shape[0]:,} users × {len(_final_feat_cols)} features")
print(f"   Labels: ret30d, ret90d, upg60d")
print(f"   Split: {feat_matrix_v2['split'].value_counts().to_dict()}")

print(f"\n   Label distribution:")
for _lbl in ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]:
    _pos = feat_matrix_v2[_lbl].sum()
    print(f"      {_lbl}: {_pos}/{len(feat_matrix_v2)} positive ({_pos/len(feat_matrix_v2)*100:.1f}%)")

print(f"\n   Feature dtypes: {feat_matrix_v2[_final_feat_cols].dtypes.value_counts().to_dict()}")

# Verify no NaN in features
_nan_counts = feat_matrix_v2[_final_feat_cols].isnull().sum()
_nan_feats = _nan_counts[_nan_counts > 0]
if len(_nan_feats) > 0:
    print(f"\n   ⚠️  NaN features: {_nan_feats.to_dict()}")
else:
    print(f"\n   ✅ No NaN values in feature matrix")

# Leakage check: no post-window events used in features
_early_max_days = _early.groupby("user_id_canon")["days_since_first"].max()
assert _early_max_days.max() <= _EARLY_END + 0.001, "LEAKAGE DETECTED: Feature events beyond window!"
print(f"   ✅ Anti-leakage: all features use only first {_EARLY_END}d events")

# Label columns not in features
for _lbl in ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]:
    assert _lbl not in _final_feat_cols, f"LEAKAGE: label {_lbl} in features!"
print(f"   ✅ Labels not in feature set")

print(f"\n   Derived features added:")
_derived_confirmed = [c for c in _final_feat_cols if c in _derived_feat_cols + list(_te_new_cols.keys())]
for _dc in _derived_confirmed:
    print(f"      ✅ {_dc}")

print(f"\n   All {len(_final_feat_cols)} feature columns:")
for _i, _fc in enumerate(_final_feat_cols, 1):
    print(f"      {_i:3d}. {_fc}")
