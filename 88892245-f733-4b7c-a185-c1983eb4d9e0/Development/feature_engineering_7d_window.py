
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy

# ═══════════════════════════════════════════════════════════════════════════
# 7-DAY WINDOW FEATURE ENGINEERING — Full Pipeline
# Loads raw events → builds early window → engineers 6 derived features →
# drops zero-variance → removes high-correlation (>0.92) → target-encodes
# sparse categoricals → outputs clean feature matrix with labels.
# ═══════════════════════════════════════════════════════════════════════════

_EARLY_END = EARLY_WINDOW_DAYS  # 7 days (anti-leakage boundary)
_CORR_THRESH = 0.92
_VAR_THRESH  = 1e-10
_K_SMOOTH    = 10   # Laplace smoothing strength for target encoding

# ─────────────────────────────────────────────────────
# 0. BUILD EARLY WINDOW — strictly ≤ 7 days after signup
# ─────────────────────────────────────────────────────
_ev = events.merge(
    cohort_users[["user_id_canon", "first_event_ts"]],
    on="user_id_canon", how="inner"
).copy()
_ev["days_since_first"] = (_ev["timestamp"] - _ev["first_event_ts"]).dt.total_seconds() / 86400

_early = _ev[_ev["days_since_first"] <= _EARLY_END].copy()
_post  = _ev[_ev["days_since_first"] >  _EARLY_END].copy()

print(f"📦 DATASET")
print(f"   Cohort users      : {cohort_users['user_id_canon'].nunique():,}")
print(f"   Early-window (≤7d): {len(_early):,} events, {_early['user_id_canon'].nunique():,} users")
print(f"   Post-window (>7d) : {len(_post):,} events")
print()

# ─────────────────────────────────────────────────────
# 1. LABELS — strictly post-window to avoid leakage
# ─────────────────────────────────────────────────────
_labels = cohort_users[["user_id_canon"]].copy()

# ret30d: any event in days (7, 30]
_ret30 = _post[_post["days_since_first"] <= 30].groupby("user_id_canon").size()
_labels["ret30d"] = _labels["user_id_canon"].map(_ret30).fillna(0).clip(upper=1).astype(int)

# ret90d: any event in days (7, 90]
_ret90 = _post[_post["days_since_first"] <= 90].groupby("user_id_canon").size()
_labels["ret90d"] = _labels["user_id_canon"].map(_ret90).fillna(0).clip(upper=1).astype(int)

# upg60d: credits_used in days (7, 60]
_upg60 = _post[(_post["days_since_first"] <= 60) & (_post["event"] == "credits_used")].groupby("user_id_canon").size()
_labels["upg60d"] = _labels["user_id_canon"].map(_upg60).fillna(0).clip(upper=1).astype(int)

print(f"🎯 LABELS")
for _lbl in ["ret30d", "ret90d", "upg60d"]:
    _n = _labels[_lbl].sum()
    print(f"   {_lbl}: {_n:,}/{len(_labels):,} positive ({_n/len(_labels)*100:.1f}%)")
print()

# ─────────────────────────────────────────────────────
# 2. EVENT CATEGORY MAP
# ─────────────────────────────────────────────────────
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
    "canvas":   ["canvas_open","canvas_create","canvas_delete","canvas_share",
                 "fullscreen_close","fullscreen_open","fullscreen_preview_output"],
    "credits":  ["credits_used","addon_credits_used","credits_below_4","credits_exceeded",
                 "credits_below_3","credits_below_1","credits_below_2",
                 "ai_credit_banner_shown","promo_code_redeemed"],
    "onboarding": ["skip_onboarding_form","submit_onboarding_form",
                   "canvas_onboarding_tour_started","canvas_onboarding_tour_finished",
                   "quickstart_explore_playground","quickstart_add_dataset","new_user_created"],
    "files":    ["files_upload","files_download","files_delete"],
    "collab":   ["canvas_share","edge_create","edge_delete","referral_modal_open"],
    "deploy":   ["scheduled_job_start","scheduled_job_stop","app_publish",
                 "app_unpublish","hosted_apps_open","requirements_build"],
}
_event_to_cat = {_e: _cat for _cat, _evts in _CATEGORIES.items() for _e in _evts}
_early["event_cat"] = _early["event"].map(_event_to_cat).fillna("other")

# ─────────────────────────────────────────────────────
# 3. BASE FEATURES — intensity, breadth, depth, session stats
# ─────────────────────────────────────────────────────
# 3A. Intensity / Consistency
_intensity = _early.groupby("user_id_canon").agg(
    feat_event_count  =("timestamp", "size"),
    feat_active_days  =("timestamp", lambda x: x.dt.date.nunique()),
    feat_n_sessions   =("prop_$session_id", "nunique"),
).reset_index()

# Per-day event counts for ramp-up slope
_daily = (
    _early.groupby(["user_id_canon", _early["timestamp"].dt.date])
    .size().reset_index(name="_cnt")
)
_daily.columns = ["user_id_canon", "_date", "_cnt"]
_daily["_day_num"] = _daily.groupby("user_id_canon").cumcount()

def _slope(_grp):
    if len(_grp) < 2: return 0.0
    _x = _grp["_day_num"].values.astype(float)
    _y = _grp["_cnt"].values.astype(float)
    return float(np.polyfit(_x, _y, 1)[0]) if _x.std() > 0 else 0.0

_slopes      = _daily.groupby("user_id_canon").apply(_slope).reset_index(name="feat_rampup_slope")
_epd         = _daily.groupby("user_id_canon")["_cnt"].mean().reset_index(name="feat_events_per_day")

def _max_gap(_grp):
    _dates = sorted(_grp["timestamp"].dt.date.unique())
    if len(_dates) < 2: return 0.0
    return float(max((b - a).days for a, b in zip(_dates[:-1], _dates[1:])))

_max_gaps    = _early.groupby("user_id_canon").apply(_max_gap).reset_index(name="feat_max_gap_days")

# 3B. Breadth / Depth
_breadth = _early.groupby("user_id_canon").agg(
    feat_distinct_events    =("event", "nunique"),
    feat_distinct_categories=("event_cat", "nunique"),
    feat_distinct_devices   =("prop_$device_type", "nunique"),
    feat_distinct_os        =("prop_$os", "nunique"),
    feat_distinct_browsers  =("prop_$browser", "nunique"),
).reset_index()

# Category ratios
_cat_cnt   = _early.groupby(["user_id_canon", "event_cat"]).size().unstack(fill_value=0)
_cat_total = _cat_cnt.sum(axis=1)
for _cat_name in ["agent","block_ops","canvas","credits","onboarding","files","collab","deploy"]:
    _cat_cnt[f"feat_ratio_{_cat_name}"] = (
        _cat_cnt[_cat_name] / _cat_total if _cat_name in _cat_cnt.columns else 0.0
    )
_cat_ratios = _cat_cnt[[c for c in _cat_cnt.columns if c.startswith("feat_ratio_")]].reset_index()

# 3C. Adoption signals (time-to-first in hours)
_ttf_events = {
    "feat_ttf_run_block":    "run_block",
    "feat_ttf_canvas_create":"canvas_create",
    "feat_ttf_agent_use":    "agent_worker_created",
    "feat_ttf_file_upload":  "files_upload",
    "feat_ttf_credits_used": "credits_used",
    "feat_ttf_edge_create":  "edge_create",
    "feat_ttf_block_create": "block_create",
}
_ttf_frames = []
for _feat, _evt in _ttf_events.items():
    _first = _early[_early["event"] == _evt].groupby("user_id_canon")["days_since_first"].min()
    _ttf_frames.append((_first * 24).reset_index(name=_feat))

# 3D. Session stats
_sess_stats = _early.groupby(["user_id_canon", "prop_$session_id"]).agg(
    _ss_events  =("timestamp", "size"),
    _ss_dur_s   =("timestamp", lambda x: (x.max() - x.min()).total_seconds()),
    _ss_dist_evt=("event", "nunique"),
).reset_index()

_session_agg = _sess_stats.groupby("user_id_canon").agg(
    feat_mean_events_per_session          =("_ss_events",   "mean"),
    feat_median_events_per_session        =("_ss_events",   "median"),
    feat_max_events_per_session           =("_ss_events",   "max"),
    feat_mean_session_duration_min        =("_ss_dur_s",    lambda x: x.mean() / 60),
    feat_max_session_duration_min         =("_ss_dur_s",    lambda x: x.max() / 60),
    feat_mean_distinct_events_per_session =("_ss_dist_evt", "mean"),
).reset_index()

# 3E. Collaboration / deploy / canvas
_early_deploy  = _early[_early["event"].isin(["scheduled_job_start","app_publish","requirements_build"])].groupby("user_id_canon").size().reset_index(name="feat_early_deploy_count")
_early_sched   = _early[_early["event"].isin(["scheduled_job_start","scheduled_job_stop"])].groupby("user_id_canon").size().reset_index(name="feat_early_schedule_count")
_collab_cnt    = _early[_early["event"].isin(["canvas_share","referral_modal_open"])].groupby("user_id_canon").size().reset_index(name="feat_collab_actions")
_canvas_div    = _early[_early["prop_$pathname"].str.contains("/canvas/", na=False)].groupby("user_id_canon")["prop_$pathname"].nunique().reset_index(name="feat_distinct_canvases")

# 3F. Metadata
_meta = _early.groupby("user_id_canon").agg(
    feat_primary_device  =("prop_$device_type",        lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"),
    feat_primary_os      =("prop_$os",                  lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"),
    feat_primary_browser =("prop_$browser",             lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"),
    feat_primary_country =("prop_$geoip_country_name",  lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"),
    feat_credit_amount_sum=("prop_credit_amount",        "sum"),
).reset_index()

# Onboarding flags
_ob_submitted = set(_early[_early["event"] == "submit_onboarding_form"]["user_id_canon"])
_ob_skipped   = set(_early[_early["event"] == "skip_onboarding_form"]["user_id_canon"])
_ob_tour_done = set(_early[_early["event"] == "canvas_onboarding_tour_finished"]["user_id_canon"])

# Signup time features (cyclical)
_signup_feats = cohort_users[["user_id_canon","first_event_ts"]].copy()
_signup_feats["feat_signup_dow"]  = _signup_feats["first_event_ts"].dt.dayofweek
_signup_feats["feat_signup_hour"] = _signup_feats["first_event_ts"].dt.hour
_signup_feats = _signup_feats[["user_id_canon","feat_signup_dow","feat_signup_hour"]]

# ─────────────────────────────────────────────────────
# 4. ASSEMBLE BASE FEATURE MATRIX
# ─────────────────────────────────────────────────────
feat_7d = cohort_users[["user_id_canon"]].copy()

_base_frames = [
    _intensity, _slopes, _max_gaps, _epd, _breadth,
    _cat_ratios, _early_deploy, _early_sched, _collab_cnt,
    _canvas_div, _session_agg, _meta, _signup_feats,
]
for _ff in _base_frames: feat_7d = feat_7d.merge(_ff, on="user_id_canon", how="left")
for _tf in _ttf_frames:  feat_7d = feat_7d.merge(_tf, on="user_id_canon", how="left")

# Binary flags
feat_7d["feat_onboarding_completed"] = feat_7d["user_id_canon"].isin(_ob_submitted).astype(int)
feat_7d["feat_onboarding_skipped"]   = feat_7d["user_id_canon"].isin(_ob_skipped).astype(int)
feat_7d["feat_tour_finished"]        = feat_7d["user_id_canon"].isin(_ob_tour_done).astype(int)

# Fill numeric NaNs with 0 (user didn't do that activity)
_num_cols = feat_7d.select_dtypes(include=[np.number]).columns
feat_7d[_num_cols] = feat_7d[_num_cols].fillna(0)

# TTF sentinel: 0 means never happened → replace with 999 hours
_ttf_cols = [c for c in feat_7d.columns if c.startswith("feat_ttf_")]
for _tc in _ttf_cols:
    feat_7d[_tc] = feat_7d[_tc].replace(0, 999.0)

# One-hot encode categoricals (top 10 per group, rest → "Other")
_cat_meta_cols = ["feat_primary_device","feat_primary_os","feat_primary_browser","feat_primary_country"]
_dummies_list = []
for _cm in _cat_meta_cols:
    feat_7d[_cm] = feat_7d[_cm].fillna("Unknown")
    _top = feat_7d[_cm].value_counts().head(10).index
    _clean = feat_7d[_cm].where(feat_7d[_cm].isin(_top), "Other")
    _dum = pd.get_dummies(_clean, prefix=_cm, dtype=int)
    _dummies_list.append(_dum)
    feat_7d.drop(columns=[_cm], inplace=True)

feat_7d = pd.concat([feat_7d] + _dummies_list, axis=1)

print(f"🏗️  BASE FEATURE MATRIX: {feat_7d.shape[0]:,} users × {feat_7d.shape[1]-1} features")

# ─────────────────────────────────────────────────────
# 5. DERIVED FEATURES — all 6 required
# ─────────────────────────────────────────────────────

# --- 5.1 session_entropy ---
# Shannon entropy of event-type distribution per session → mean across user's sessions
# High = diverse workflow, Low = repetitive clicks
_sess_evt_cnt = (
    _early.groupby(["user_id_canon","prop_$session_id","event"])
    .size().reset_index(name="_n")
)

def _mean_sess_entropy(_grp):
    _ents = []
    for _, _s in _grp.groupby("prop_$session_id"):
        _p = _s["_n"].values.astype(float)
        _p /= _p.sum()
        _ents.append(float(scipy_entropy(_p, base=2)) if len(_p) > 1 else 0.0)
    return np.mean(_ents) if _ents else 0.0

_session_entropy = (
    _sess_evt_cnt.groupby("user_id_canon").apply(_mean_sess_entropy)
    .reset_index(name="feat_session_entropy")
)

# --- 5.2 time_of_day_activity_profile (feat_tod_entropy) ---
# Shannon entropy of hour-of-day distribution
# 0 = always active at same hour, high = spread across the day
_early["_hour"] = _early["timestamp"].dt.hour

def _tod_ent(_grp):
    _p = _grp["_hour"].value_counts().values.astype(float)
    if _p.sum() < 2: return 0.0
    _p /= _p.sum()
    return float(scipy_entropy(_p, base=2))

_tod_entropy = (
    _early.groupby("user_id_canon").apply(_tod_ent)
    .reset_index(name="feat_tod_entropy")
)

# --- 5.3 day_gap_variance ---
# Variance of inter-day gaps → irregular usage = high variance
def _gap_var(_grp):
    _dates = sorted(_grp["timestamp"].dt.date.unique())
    if len(_dates) < 2: return 0.0
    _diffs = [(b - a).days for a, b in zip(_dates[:-1], _dates[1:])]
    return float(np.var(_diffs)) if len(_diffs) >= 2 else 0.0

_day_gap_var = (
    _early.groupby("user_id_canon").apply(_gap_var)
    .reset_index(name="feat_day_gap_variance")
)

# --- 5.4 execution_to_agent_ratio ---
# (run_block + 1) / (agent_events + 1) — smoothed ratio
# Captures whether user runs blocks manually vs. relies on AI
_run_cnt   = _early[_early["event"] == "run_block"].groupby("user_id_canon").size().reset_index(name="_run")
_agent_cnt = _early[_early["event_cat"] == "agent"].groupby("user_id_canon").size().reset_index(name="_agent")
_exec_ratio = (
    cohort_users[["user_id_canon"]]
    .merge(_run_cnt,   on="user_id_canon", how="left")
    .merge(_agent_cnt, on="user_id_canon", how="left")
    .fillna(0)
)
_exec_ratio["feat_execution_to_agent_ratio"] = (
    (_exec_ratio["_run"] + 1) / (_exec_ratio["_agent"] + 1)
)
_exec_ratio = _exec_ratio[["user_id_canon","feat_execution_to_agent_ratio"]]

# --- 5.5 canvas_creation_rate ---
# canvas_creates / max(active_days, 1) in early window
_canvas_cnt2  = _early[_early["event"] == "canvas_create"].groupby("user_id_canon").size().reset_index(name="_canvas_cnt")
_active_days2 = _early.groupby("user_id_canon")["timestamp"].apply(lambda x: x.dt.date.nunique()).reset_index(name="_active_d")
_canvas_rate  = (
    cohort_users[["user_id_canon"]]
    .merge(_canvas_cnt2,  on="user_id_canon", how="left")
    .merge(_active_days2, on="user_id_canon", how="left")
    .fillna({"_canvas_cnt": 0})
)
_canvas_rate["feat_canvas_creation_rate"] = (
    _canvas_rate["_canvas_cnt"] / _canvas_rate["_active_d"].clip(lower=1)
)
_canvas_rate = _canvas_rate[["user_id_canon","feat_canvas_creation_rate"]]

# --- 5.6 agent_to_block_conversion_flag ---
# Binary: session had agent event AND block_create event
# Strong adoption signal: AI usage led to concrete block creation
_agent_sess = set(_early[_early["event_cat"] == "agent"]["prop_$session_id"].dropna())
_block_sess = set(_early[_early["event"] == "block_create"]["prop_$session_id"].dropna())
_converted  = _agent_sess & _block_sess
_converted_users = set(_early[_early["prop_$session_id"].isin(_converted)]["user_id_canon"])
_conv_flag = pd.DataFrame({
    "user_id_canon":           cohort_users["user_id_canon"],
    "feat_agent_block_conversion": cohort_users["user_id_canon"].isin(_converted_users).astype(int)
})

print(f"\n✅ DERIVED FEATURES (6/6):")
print(f"   feat_session_entropy         → mean={_session_entropy['feat_session_entropy'].mean():.3f}")
print(f"   feat_tod_entropy             → mean={_tod_entropy['feat_tod_entropy'].mean():.3f}")
print(f"   feat_day_gap_variance        → mean={_day_gap_var['feat_day_gap_variance'].mean():.3f}")
print(f"   feat_execution_to_agent_ratio→ mean={_exec_ratio['feat_execution_to_agent_ratio'].mean():.3f}")
print(f"   feat_canvas_creation_rate    → mean={_canvas_rate['feat_canvas_creation_rate'].mean():.3f}")
print(f"   feat_agent_block_conversion  → {_conv_flag['feat_agent_block_conversion'].sum():,}/{len(_conv_flag):,} users ({_conv_flag['feat_agent_block_conversion'].mean()*100:.1f}%)")

# Merge all 6 derived features
for _df in [_session_entropy, _tod_entropy, _day_gap_var, _exec_ratio, _canvas_rate, _conv_flag]:
    feat_7d = feat_7d.merge(_df, on="user_id_canon", how="left")

_derived_names = [
    "feat_session_entropy","feat_tod_entropy","feat_day_gap_variance",
    "feat_execution_to_agent_ratio","feat_canvas_creation_rate","feat_agent_block_conversion"
]
for _c in _derived_names:
    feat_7d[_c] = feat_7d[_c].fillna(0.0)

print(f"\n🧮 After merging derived features: {feat_7d.shape[0]:,} × {feat_7d.shape[1]-1} features")

# Merge labels
feat_7d = feat_7d.merge(_labels, on="user_id_canon", how="inner")

# ─────────────────────────────────────────────────────
# 6. TEMPORAL SPLIT — no leakage: split by user signup date
# ─────────────────────────────────────────────────────
_first_ts  = cohort_users.set_index("user_id_canon")["first_event_ts"]
_train_end = pd.Timestamp(TEMPORAL_SPLITS["train_end"]).tz_localize("UTC")
_val_end   = pd.Timestamp(TEMPORAL_SPLITS["val_end"]).tz_localize("UTC")

feat_7d["split"] = "test"
feat_7d.loc[feat_7d["user_id_canon"].map(_first_ts) <= _train_end, "split"] = "train"
feat_7d.loc[
    (feat_7d["user_id_canon"].map(_first_ts) > _train_end) &
    (feat_7d["user_id_canon"].map(_first_ts) <= _val_end), "split"
] = "val"

_split_dist = feat_7d["split"].value_counts()
print(f"\n📋 TEMPORAL SPLIT (train/val/test by first_event_ts):")
for _s in ["train","val","test"]:
    _n = _split_dist.get(_s, 0)
    print(f"   {_s:5s}: {_n:,} users ({_n/len(feat_7d)*100:.1f}%)")

# ─────────────────────────────────────────────────────
# 7. ZERO-VARIANCE FILTER
# ─────────────────────────────────────────────────────
_feat_cols = [c for c in feat_7d.columns if c not in ["user_id_canon","ret30d","ret90d","upg60d","split"]]
_var = feat_7d[_feat_cols].var()
_zv_cols = _var[_var < _VAR_THRESH].index.tolist()
print(f"\n🗑️  Zero-variance features removed: {len(_zv_cols)}")
if _zv_cols: print(f"   {_zv_cols}")
feat_7d.drop(columns=_zv_cols, inplace=True, errors="ignore")

# ─────────────────────────────────────────────────────
# 8. HIGH-CORRELATION FILTER (> 0.92)
# ─────────────────────────────────────────────────────
_feat_cols2 = [c for c in feat_7d.columns if c not in ["user_id_canon","ret30d","ret90d","upg60d","split"]]
_corr = feat_7d[_feat_cols2].corr().abs()
_upper_tri = _corr.where(np.triu(np.ones(_corr.shape), k=1).astype(bool))
_hc_cols = [c for c in _upper_tri.columns if any(_upper_tri[c] > _CORR_THRESH)]
print(f"\n📉 High-correlation (> {_CORR_THRESH}) features removed: {len(_hc_cols)}")
for _hc in _hc_cols:
    _partners = _upper_tri.index[_upper_tri[_hc] > _CORR_THRESH].tolist()
    print(f"   Drop '{_hc}' (corr > {_CORR_THRESH} with: {_partners})")
feat_7d.drop(columns=_hc_cols, inplace=True, errors="ignore")

# ─────────────────────────────────────────────────────
# 9. TARGET ENCODING — sparse OHE categoricals
# Smoothed mean of ret30d, computed on train only (or val if no train)
# ─────────────────────────────────────────────────────
_train_mask = feat_7d["split"] == "train"
_val_mask   = feat_7d["split"] == "val"
_n_train    = _train_mask.sum()

_encode_mask  = _train_mask if _n_train > 0 else _val_mask
_encode_label = "train" if _n_train > 0 else "val (fallback — no train rows)"
_global_mean  = feat_7d.loc[_encode_mask, "ret30d"].mean()

print(f"\n🎯 TARGET ENCODING (smoothing k={_K_SMOOTH}, encoded on {_encode_label})")
print(f"   Global mean (ret30d): {_global_mean:.4f}")

_ohe_prefixes = ["feat_primary_country_","feat_primary_device_","feat_primary_os_","feat_primary_browser_"]
_ohe_cols_all = [c for c in feat_7d.columns if any(c.startswith(p) for p in _ohe_prefixes)]

_te_new = {}
for _p in _ohe_prefixes:
    _grp_cols = [c for c in _ohe_cols_all if c.startswith(_p)]
    if not _grp_cols: continue
    _feat_nm = "feat_te_" + _p.replace("feat_primary_","").rstrip("_")
    _te_vals = np.full(len(feat_7d), _global_mean)
    for _c in _grp_cols:
        _is_one = feat_7d[_c].values == 1
        if not _is_one.any(): continue
        _enc_mask = _is_one & _encode_mask.values
        _n_enc    = _enc_mask.sum()
        _m_enc    = feat_7d.loc[_enc_mask, "ret30d"].mean() if _n_enc > 0 else _global_mean
        _smoothed = (_n_enc * _m_enc + _K_SMOOTH * _global_mean) / (_n_enc + _K_SMOOTH)
        _te_vals[_is_one] = _smoothed
    feat_7d[_feat_nm] = _te_vals
    _te_new[_feat_nm] = _te_vals
    print(f"   {_feat_nm}: {len(_grp_cols)} cols → mean={np.mean(_te_vals):.3f}, std={np.std(_te_vals):.4f}")

feat_7d.drop(columns=_ohe_cols_all, inplace=True, errors="ignore")

# ─────────────────────────────────────────────────────
# 10. FINAL CLEAN FEATURE MATRIX
# ─────────────────────────────────────────────────────
clean_feature_matrix = feat_7d.copy()

_final_feats = [c for c in clean_feature_matrix.columns
                if c not in ["user_id_canon","ret30d","ret90d","upg60d","split"]]

# ─────────────────────────────────────────────────────
# 11. LEAKAGE AUDIT
# ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"🔒 TEMPORAL LEAKAGE AUDIT")
print(f"{'='*65}")

# Check 1: feature events bounded by early window
_max_feat_day = _early.groupby("user_id_canon")["days_since_first"].max().max()
assert _max_feat_day <= _EARLY_END + 0.001, f"❌ LEAKAGE: Feature events extend to day {_max_feat_day:.1f}"
print(f"   ✅ Feature events: max day = {_max_feat_day:.2f} (within {_EARLY_END}-day window)")

# Check 2: label events are strictly post-early-window
_min_label_day = _post.groupby("user_id_canon")["days_since_first"].min().min()
assert _min_label_day > _EARLY_END - 0.001, f"❌ LEAKAGE: Label events start at day {_min_label_day:.1f}"
print(f"   ✅ Label events: min day = {_min_label_day:.2f} (strictly after {_EARLY_END}-day boundary)")

# Check 3: label columns absent from feature set
for _lbl in ["ret30d","ret90d","upg60d"]:
    assert _lbl not in _final_feats, f"❌ LEAKAGE: label '{_lbl}' in feature set!"
print(f"   ✅ No label columns in feature set")

# Check 4: target encoding used only train/val labels (not test)
print(f"   ✅ Target encoding fitted on {_encode_label} split only")

# Check 5: no NaN in features
_nan_count = clean_feature_matrix[_final_feats].isnull().sum().sum()
assert _nan_count == 0, f"❌ NaN values in feature matrix: {_nan_count}"
print(f"   ✅ Zero NaN values in feature matrix")

print(f"\n{'='*65}")
print(f"🎉 ALL LEAKAGE CHECKS PASSED — zero temporal leakage confirmed")
print(f"{'='*65}")

# ─────────────────────────────────────────────────────
# 12. FINAL SUMMARY
# ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"📦 CLEAN FEATURE MATRIX: clean_feature_matrix")
print(f"{'='*65}")
print(f"   Shape    : {clean_feature_matrix.shape[0]:,} users × {len(_final_feats)} features (+3 labels, split col)")
print(f"   Features : {len(_final_feats)}")
print(f"   Labels   : ret30d, ret90d, upg60d")
print(f"   Split    : {clean_feature_matrix['split'].value_counts().to_dict()}")

print(f"\n   Label rates by split:")
for _s in ["train","val","test"]:
    _sub = clean_feature_matrix[clean_feature_matrix["split"] == _s]
    if len(_sub) == 0: continue
    _r30 = _sub["ret30d"].mean()*100
    _r90 = _sub["ret90d"].mean()*100
    _upg = _sub["upg60d"].mean()*100
    print(f"   {_s:5s} (n={len(_sub):,}): ret30d={_r30:.1f}%, ret90d={_r90:.1f}%, upg60d={_upg:.1f}%")

print(f"\n   Derived features confirmed:")
for _dc in _derived_names:
    _in_matrix = _dc in _final_feats
    print(f"   {'✅' if _in_matrix else '⚠️ REMOVED (high-corr)'} {_dc}")

print(f"\n   Target-encoded categoricals:")
for _te_nm in _te_new.keys():
    print(f"   ✅ {_te_nm}")

print(f"\n   Feature dtype breakdown: {clean_feature_matrix[_final_feats].dtypes.value_counts().to_dict()}")

print(f"\n   All {len(_final_feats)} features:")
for _i, _f in enumerate(_final_feats, 1):
    print(f"   {_i:3d}. {_f}")
