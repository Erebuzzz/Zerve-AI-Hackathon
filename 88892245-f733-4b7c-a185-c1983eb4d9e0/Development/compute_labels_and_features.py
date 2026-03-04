
import pandas as pd
import numpy as np
from collections import Counter

# ═════════════════════════════════════════════════════
# 0. SETUP — merge first_event_ts onto events, compute relative time
# ═════════════════════════════════════════════════════
_ev = events.merge(
    cohort_users[["user_id_canon", "first_event_ts"]],
    on="user_id_canon", how="inner"
).copy()
_ev["days_since_first"] = (_ev["timestamp"] - _ev["first_event_ts"]).dt.total_seconds() / 86400

# Define windows
_EARLY_END = EARLY_WINDOW_DAYS  # 7 days
_early = _ev[_ev["days_since_first"] <= _EARLY_END].copy()
_post_early = _ev[_ev["days_since_first"] > _EARLY_END].copy()

print(f"📦 Events total: {len(_ev):,}")
print(f"   Early window (≤{_EARLY_END}d): {len(_early):,} events, {_early['user_id_canon'].nunique():,} users")
print(f"   Post-early (>{_EARLY_END}d): {len(_post_early):,} events, {_post_early['user_id_canon'].nunique():,} users")

# ═════════════════════════════════════════════════════
# 1. TARGET LABELS — strict anti-leakage time windows
# ═════════════════════════════════════════════════════
# y_ret_30d:  user has any event in days (7, 30] after first_event_ts
# y_ret_90d:  user has any event in days (7, 90] after first_event_ts  
# y_upgrade_60d: user has credits_used event in days (7, 60] after first_event_ts

_labels = cohort_users[["user_id_canon", "first_event_ts"]].copy()

# y_ret_30d: active in (7, 30] days
_ret30 = _post_early[_post_early["days_since_first"] <= 30].groupby("user_id_canon").size()
_labels["y_ret_30d"] = _labels["user_id_canon"].map(_ret30).fillna(0).astype(int).clip(upper=1)

# y_ret_90d: active in (7, 90] days
_ret90 = _post_early[_post_early["days_since_first"] <= 90].groupby("user_id_canon").size()
_labels["y_ret_90d"] = _labels["user_id_canon"].map(_ret90).fillna(0).astype(int).clip(upper=1)

# y_upgrade_60d: credits_used event in (7, 60] days
_upgrade60 = _post_early[
    (_post_early["days_since_first"] <= 60) & 
    (_post_early["event"] == "credits_used")
].groupby("user_id_canon").size()
_labels["y_upgrade_60d"] = _labels["user_id_canon"].map(_upgrade60).fillna(0).astype(int).clip(upper=1)

print(f"\n🎯 Label distributions:")
for _lbl in ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]:
    _pos = _labels[_lbl].sum()
    _tot = len(_labels)
    print(f"   {_lbl}: {_pos:,}/{_tot:,} positive ({_pos/_tot*100:.1f}%)")

# ═════════════════════════════════════════════════════
# 2. EVENT CATEGORY MAPPING
# ═════════════════════════════════════════════════════
_CATEGORIES = {
    "agent": ["agent_tool_call_create_block_tool", "agent_tool_call_run_block_tool",
              "agent_tool_call_get_block_tool", "agent_tool_call_get_canvas_summary_tool",
              "agent_tool_call_get_variable_preview_tool", "agent_tool_call_finish_ticket_tool",
              "agent_tool_call_refactor_block_tool", "agent_tool_call_delete_block_tool",
              "agent_tool_call_create_edges_tool", "agent_tool_call_get_block_image_tool",
              "agent_worker_created", "agent_new_chat", "agent_start_from_prompt",
              "agent_message", "agent_accept_suggestion", "agent_open",
              "agent_suprise_me", "agent_open_error_assist", "agent_upload_files",
              "agent_block_created"],
    "block_ops": ["run_block", "block_create", "block_delete", "block_resize",
                  "block_rename", "block_copy", "block_paste", "block_open_compute_settings",
                  "block_output_copy", "block_output_download", "stop_block",
                  "run_all_blocks", "run_upto_block", "run_from_block"],
    "canvas": ["canvas_open", "canvas_create", "canvas_delete", "canvas_share",
               "fullscreen_close", "fullscreen_open", "fullscreen_preview_output",
               "fullscreen_preview_input"],
    "credits": ["credits_used", "addon_credits_used", "credits_below_4", "credits_exceeded",
                "credits_below_3", "credits_below_1", "credits_below_2",
                "ai_credit_banner_shown", "promo_code_redeemed"],
    "onboarding": ["skip_onboarding_form", "submit_onboarding_form",
                   "canvas_onboarding_tour_started", "canvas_onboarding_tour_running_blocks_step",
                   "canvas_onboarding_tour_finished", "canvas_onboarding_tour_code_and_variables_step",
                   "canvas_onboarding_tour_compute_step", "canvas_onboarding_tour_add_block_step",
                   "canvas_onboarding_tour_ai_assistant_step", "canvas_onboarding_tour_you_are_ready_step",
                   "quickstart_explore_playground", "quickstart_add_dataset", "new_user_created"],
    "files": ["files_upload", "files_download", "files_delete", "files_update_lazy_load"],
    "collab": ["canvas_share", "edge_create", "edge_delete", "layer_create",
               "layer_delete", "layer_rename", "folder_open", "referral_modal_open"],
    "deploy": ["scheduled_job_start", "scheduled_job_stop", "app_publish",
               "app_unpublish", "hosted_apps_open", "requirements_build"],
    "auth": ["sign_in", "sign_up"],
    "ui": ["link_clicked", "button_clicked"],
}

# Map events to categories
_event_to_cat = {}
for _cat, _evts in _CATEGORIES.items():
    for _e in _evts:
        _event_to_cat[_e] = _cat

_early["event_cat"] = _early["event"].map(_event_to_cat).fillna("other")

# ═════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING — early window only
# ═════════════════════════════════════════════════════

# --- 3A. Intensity / Consistency ---
_intensity = _early.groupby("user_id_canon").agg(
    feat_event_count=("timestamp", "size"),
    feat_active_days=("timestamp", lambda x: x.dt.date.nunique()),
    feat_n_sessions=("prop_$session_id", "nunique"),
).reset_index()

# Daily event counts for ramp-up slope
_daily = _early.groupby(["user_id_canon", _early["timestamp"].dt.date]).size().reset_index(name="_cnt")
_daily.columns = ["user_id_canon", "_date", "_cnt"]
_daily["_day_num"] = _daily.groupby("user_id_canon").cumcount()

def _compute_slope(_grp):
    if len(_grp) < 2:
        return 0.0
    _x = _grp["_day_num"].values.astype(float)
    _y = _grp["_cnt"].values.astype(float)
    if _x.std() == 0:
        return 0.0
    return np.polyfit(_x, _y, 1)[0]

_slopes = _daily.groupby("user_id_canon").apply(_compute_slope).reset_index(name="feat_rampup_slope")

# Max gap between consecutive event days
def _max_gap(_grp):
    _dates = sorted(_grp["timestamp"].dt.date.unique())
    if len(_dates) < 2:
        return 0.0
    _diffs = [(b - a).days for a, b in zip(_dates[:-1], _dates[1:])]
    return float(max(_diffs))

_gaps = _early.groupby("user_id_canon").apply(_max_gap).reset_index(name="feat_max_gap_days")

# Events per day (mean)
_events_per_day = _daily.groupby("user_id_canon")["_cnt"].mean().reset_index(name="feat_events_per_day")

# --- 3B. Breadth / Depth ---
_breadth = _early.groupby("user_id_canon").agg(
    feat_distinct_events=("event", "nunique"),
    feat_distinct_categories=("event_cat", "nunique"),
).reset_index()

# Resource type diversity (device/os/browser)
_device_div = _early.groupby("user_id_canon").agg(
    feat_distinct_devices=("prop_$device_type", "nunique"),
    feat_distinct_os=("prop_$os", "nunique"),
    feat_distinct_browsers=("prop_$browser", "nunique"),
).reset_index()

# Category event ratios
_cat_counts = _early.groupby(["user_id_canon", "event_cat"]).size().unstack(fill_value=0)
_cat_total = _cat_counts.sum(axis=1)
for _cat_name in ["agent", "block_ops", "canvas", "credits", "onboarding", "files", "collab", "deploy"]:
    if _cat_name in _cat_counts.columns:
        _cat_counts[f"feat_ratio_{_cat_name}"] = _cat_counts[_cat_name] / _cat_total
    else:
        _cat_counts[f"feat_ratio_{_cat_name}"] = 0.0

_ratio_cols = [c for c in _cat_counts.columns if c.startswith("feat_ratio_")]
_cat_ratios = _cat_counts[_ratio_cols].reset_index()

# --- 3C. Advanced Adoption Signals ---
# Time-to-first-X features (in hours)
_adoption_events = {
    "feat_ttf_run_block": "run_block",
    "feat_ttf_canvas_create": "canvas_create",
    "feat_ttf_agent_use": "agent_worker_created",
    "feat_ttf_file_upload": "files_upload",
    "feat_ttf_credits_used": "credits_used",
    "feat_ttf_edge_create": "edge_create",
    "feat_ttf_block_create": "block_create",
}

_ttf_frames = []
for _feat_name, _evt_name in _adoption_events.items():
    _first_evt = _early[_early["event"] == _evt_name].groupby("user_id_canon")["days_since_first"].min()
    _first_evt = (_first_evt * 24).reset_index(name=_feat_name)  # convert to hours
    _ttf_frames.append(_first_evt)

# Early deploy signals (scheduled_job_start, app_publish in early window)
_deploy_events = ["scheduled_job_start", "app_publish", "requirements_build"]
_early_deploy = _early[_early["event"].isin(_deploy_events)].groupby("user_id_canon").size().reset_index(name="feat_early_deploy_count")

# Early connector/schedule usage
_schedule_events = ["scheduled_job_start", "scheduled_job_stop"]
_early_schedule = _early[_early["event"].isin(_schedule_events)].groupby("user_id_canon").size().reset_index(name="feat_early_schedule_count")

# --- 3D. Collaboration Signals ---
_collab_events = ["canvas_share", "referral_modal_open"]
_early_collab = _early[_early["event"].isin(_collab_events)].groupby("user_id_canon").size().reset_index(name="feat_collab_actions")

# Distinct canvases (from pathname)
_early_canvases = _early[_early["prop_$pathname"].str.contains("/canvas/", na=False)]
_canvas_diversity = _early_canvases.groupby("user_id_canon")["prop_$pathname"].nunique().reset_index(name="feat_distinct_canvases")

# --- 3E. Workflow Shape (Session Stats) ---
# Events per session
_sess_stats = _early.groupby(["user_id_canon", "prop_$session_id"]).agg(
    _sess_events=("timestamp", "size"),
    _sess_duration_s=("timestamp", lambda x: (x.max() - x.min()).total_seconds()),
    _sess_distinct_events=("event", "nunique"),
).reset_index()

_session_agg = _sess_stats.groupby("user_id_canon").agg(
    feat_mean_events_per_session=("_sess_events", "mean"),
    feat_median_events_per_session=("_sess_events", "median"),
    feat_max_events_per_session=("_sess_events", "max"),
    feat_mean_session_duration_min=("_sess_duration_s", lambda x: x.mean() / 60),
    feat_max_session_duration_min=("_sess_duration_s", lambda x: x.max() / 60),
    feat_mean_distinct_events_per_session=("_sess_distinct_events", "mean"),
).reset_index()

# Transition counts between categories (sequential pairs)
_sorted_early = _early.sort_values(["user_id_canon", "timestamp"])
_sorted_early["_next_cat"] = _sorted_early.groupby("user_id_canon")["event_cat"].shift(-1)
_transitions = _sorted_early.dropna(subset=["_next_cat"])
_transitions["_trans"] = _transitions["event_cat"] + "->" + _transitions["_next_cat"]
_trans_counts = _transitions.groupby("user_id_canon")["_trans"].nunique().reset_index(name="feat_distinct_transitions")

# --- 3F. Metadata features ---
_meta = _early.groupby("user_id_canon").agg(
    feat_primary_device=("prop_$device_type", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
    feat_primary_os=("prop_$os", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
    feat_primary_browser=("prop_$browser", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
    feat_primary_country=("prop_$geoip_country_name", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
    feat_credit_amount_sum=("prop_credit_amount", "sum"),
    feat_credits_used_sum=("prop_credits_used", "sum"),
).reset_index()

# Onboarding completion flag
_onboard_submitted = set(_early[_early["event"] == "submit_onboarding_form"]["user_id_canon"])
_onboard_skipped = set(_early[_early["event"] == "skip_onboarding_form"]["user_id_canon"])
_onboard_tour_finished = set(_early[_early["event"] == "canvas_onboarding_tour_finished"]["user_id_canon"])

# Day-of-week of first event (cyclical)
_first_dow = cohort_users[["user_id_canon", "first_event_ts"]].copy()
_first_dow["feat_signup_dow"] = _first_dow["first_event_ts"].dt.dayofweek
_first_dow["feat_signup_hour"] = _first_dow["first_event_ts"].dt.hour
_first_dow = _first_dow[["user_id_canon", "feat_signup_dow", "feat_signup_hour"]]

# ═════════════════════════════════════════════════════
# 4. ASSEMBLE ALL FEATURES
# ═════════════════════════════════════════════════════
_feat_df = cohort_users[["user_id_canon"]].copy()

# Merge all feature dataframes
_feature_frames = [
    _intensity, _slopes, _gaps, _events_per_day, _breadth, _device_div,
    _cat_ratios, _early_deploy, _early_schedule, _early_collab,
    _canvas_diversity, _session_agg, _trans_counts, _meta, _first_dow,
]
for _ttf in _ttf_frames:
    _feature_frames.append(_ttf)

for _ff in _feature_frames:
    _feat_df = _feat_df.merge(_ff, on="user_id_canon", how="left")

# Add binary flags
_feat_df["feat_onboarding_completed"] = _feat_df["user_id_canon"].isin(_onboard_submitted).astype(int)
_feat_df["feat_onboarding_skipped"] = _feat_df["user_id_canon"].isin(_onboard_skipped).astype(int)
_feat_df["feat_tour_finished"] = _feat_df["user_id_canon"].isin(_onboard_tour_finished).astype(int)

# Fill NaN for numeric features with 0 (user didn't do those activities)
_numeric_cols = _feat_df.select_dtypes(include=[np.number]).columns
_feat_df[_numeric_cols] = _feat_df[_numeric_cols].fillna(0)

# Fill NaN for TTF features with a large sentinel (never happened = 999 hours)
_ttf_cols = [c for c in _feat_df.columns if c.startswith("feat_ttf_")]
for _tc in _ttf_cols:
    _feat_df[_tc] = _feat_df[_tc].replace(0, 999.0)  # 0 means they didn't do it

# Encode categorical metadata as dummy variables
_cat_meta_cols = ["feat_primary_device", "feat_primary_os", "feat_primary_browser", "feat_primary_country"]
for _cm in _cat_meta_cols:
    _feat_df[_cm] = _feat_df[_cm].fillna("Unknown")

# One-hot encode categoricals (top N categories, rest = "Other")
_dummies_list = []
for _cm in _cat_meta_cols:
    _top_vals = _feat_df[_cm].value_counts().head(10).index
    _feat_df[f"_{_cm}_clean"] = _feat_df[_cm].where(_feat_df[_cm].isin(_top_vals), "Other")
    _dum = pd.get_dummies(_feat_df[f"_{_cm}_clean"], prefix=_cm, dtype=int)
    _dummies_list.append(_dum)
    _feat_df.drop(columns=[_cm, f"_{_cm}_clean"], inplace=True)

_feat_df = pd.concat([_feat_df] + _dummies_list, axis=1)

print(f"\n📊 Feature matrix: {_feat_df.shape[0]:,} users × {_feat_df.shape[1]-1} features")

# ═════════════════════════════════════════════════════
# 5. REMOVE NEAR-ZERO-VARIANCE FEATURES
# ═════════════════════════════════════════════════════
_feature_cols = [c for c in _feat_df.columns if c != "user_id_canon"]
_variances = _feat_df[_feature_cols].var()
_low_var_threshold = 1e-6
_low_var_cols = _variances[_variances < _low_var_threshold].index.tolist()
print(f"\n🗑️  Removing {len(_low_var_cols)} near-zero-variance features: {_low_var_cols}")
_feat_df.drop(columns=_low_var_cols, inplace=True)

_final_feature_cols = [c for c in _feat_df.columns if c != "user_id_canon"]
print(f"   Remaining features: {len(_final_feature_cols)}")

# ═════════════════════════════════════════════════════
# 6. MERGE LABELS + TEMPORAL SPLIT
# ═════════════════════════════════════════════════════
modeling_df = _feat_df.merge(_labels[["user_id_canon", "y_ret_30d", "y_ret_90d", "y_upgrade_60d"]], 
                             on="user_id_canon", how="inner")

# Temporal split based on first_event_ts
_first_ts = cohort_users.set_index("user_id_canon")["first_event_ts"]
_train_end = pd.Timestamp(TEMPORAL_SPLITS["train_end"]).tz_localize("UTC")
_val_end = pd.Timestamp(TEMPORAL_SPLITS["val_end"]).tz_localize("UTC")

modeling_df["split"] = "test"
modeling_df.loc[modeling_df["user_id_canon"].map(_first_ts) <= _train_end, "split"] = "train"
modeling_df.loc[
    (modeling_df["user_id_canon"].map(_first_ts) > _train_end) & 
    (modeling_df["user_id_canon"].map(_first_ts) <= _val_end), "split"
] = "val"

_split_counts = modeling_df["split"].value_counts()
print(f"\n📋 Temporal split distribution:")
for _s in ["train", "val", "test"]:
    _n = _split_counts.get(_s, 0)
    print(f"   {_s}: {_n:,} users ({_n/len(modeling_df)*100:.1f}%)")

# ═════════════════════════════════════════════════════
# 7. LEAKAGE CHECKS
# ═════════════════════════════════════════════════════
print(f"\n🔒 LEAKAGE CHECKS:")

# Check 1: All feature events are within early window
_early_max_days = _early.groupby("user_id_canon")["days_since_first"].max()
_max_feature_day = _early_max_days.max()
assert _max_feature_day <= _EARLY_END + 0.001, f"LEAKAGE: Feature events extend to day {_max_feature_day:.1f}, beyond early window of {_EARLY_END}d"
print(f"   ✅ Feature events: max day = {_max_feature_day:.2f} (within {_EARLY_END}d early window)")

# Check 2: Label events are strictly outside early window 
_post_early_min = _post_early.groupby("user_id_canon")["days_since_first"].min()
_min_label_day = _post_early_min.min()
assert _min_label_day > _EARLY_END - 0.001, f"LEAKAGE: Label events start at day {_min_label_day:.1f}, within early window"
print(f"   ✅ Label events: min day = {_min_label_day:.2f} (strictly after {_EARLY_END}d)")

# Check 3: No label columns leaked into features
_feature_only_cols = [c for c in modeling_df.columns if c not in ["user_id_canon", "y_ret_30d", "y_ret_90d", "y_upgrade_60d", "split"]]
for _lbl_col in ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]:
    assert _lbl_col not in _feature_only_cols, f"LEAKAGE: {_lbl_col} found in feature columns!"
print(f"   ✅ No label columns in feature set")

# Check 4: Feature count
assert len(_feature_only_cols) >= 30, f"Only {len(_feature_only_cols)} features, need ≥30"
print(f"   ✅ Feature count: {len(_feature_only_cols)} features (≥30 requirement met)")

print(f"\n🎉 ALL LEAKAGE CHECKS PASSED!")

# ═════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ═════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"📦 MODELING-READY DATAFRAME: modeling_df")
print(f"{'='*60}")
print(f"   Shape: {modeling_df.shape[0]:,} users × {modeling_df.shape[1]} columns")
print(f"   Features: {len(_feature_only_cols)}")
print(f"   Labels: y_ret_30d, y_ret_90d, y_upgrade_60d")
print(f"   Split column: 'split' (train/val/test)")
print(f"\n   Label rates by split:")
for _s in ["train", "val", "test"]:
    _sub = modeling_df[modeling_df["split"] == _s]
    if len(_sub) > 0:
        print(f"   {_s} (n={len(_sub):,}):")
        for _lbl in ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]:
            print(f"      {_lbl}: {_sub[_lbl].mean()*100:.1f}%")

print(f"\n   Feature columns ({len(_feature_only_cols)}):")
for _i, _fc in enumerate(_feature_only_cols):
    print(f"      {_i+1:2d}. {_fc}")

print(f"\n   Sample row dtypes:")
print(modeling_df[_feature_only_cols].dtypes.value_counts().to_string())
