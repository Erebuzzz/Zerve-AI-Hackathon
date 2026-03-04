
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────
# LOAD & PARSE
# ─────────────────────────────────────────────────────
_raw = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv", low_memory=False)

# Parse timestamp — mixed formats (some with/without microseconds)
_raw["timestamp"] = pd.to_datetime(_raw["timestamp"], format="ISO8601", utc=True)

# Use person_id as canonical user key (fall back to distinct_id if missing)
_raw["user_id_canon"] = _raw["person_id"].fillna(_raw["distinct_id"])

# Keep useful columns for downstream — trim the 107-col monster
_keep_cols = [
    "user_id_canon", "event", "timestamp",
    "prop_$pathname", "prop_$device_type", "prop_$os", "prop_$browser",
    "prop_$geoip_country_name", "prop_$geoip_country_code",
    "prop_surface", "prop_tool_name", "prop_credit_amount",
    "prop_credits_used", "prop_$session_id",
]
_keep_cols = [c for c in _keep_cols if c in _raw.columns]
events_all = _raw[_keep_cols].copy()

print(f"📦 Parsed events: {events_all.shape[0]:,} rows, {events_all['user_id_canon'].nunique():,} unique users")
print(f"   Timestamp range: {events_all['timestamp'].min()} → {events_all['timestamp'].max()}")

# ─────────────────────────────────────────────────────
# EVENT TAXONOMY
# ─────────────────────────────────────────────────────
_event_counts = events_all["event"].value_counts()
print(f"\n🏷️  Event types ({len(_event_counts)} unique):")
print(_event_counts.to_string())

# ─────────────────────────────────────────────────────
# PER-USER TIMELINES
# ─────────────────────────────────────────────────────
_user_agg = events_all.groupby("user_id_canon").agg(
    first_event_ts=("timestamp", "min"),
    last_event_ts=("timestamp", "max"),
    total_events=("timestamp", "size"),
    distinct_days=("timestamp", lambda x: x.dt.date.nunique()),
    n_sessions=("prop_$session_id", "nunique"),
).reset_index()

_user_agg["active_span_days"] = (_user_agg["last_event_ts"] - _user_agg["first_event_ts"]).dt.total_seconds() / 86400

print(f"\n👤 User timelines computed for {len(_user_agg):,} users")
print(_user_agg[["total_events", "distinct_days", "active_span_days", "n_sessions"]].describe().round(2).to_string())

# ─────────────────────────────────────────────────────
# COHORT FILTER
# ─────────────────────────────────────────────────────
_cutoff = pd.Timestamp(COHORT_CUTOFF).tz_localize("UTC")
cohort_users = _user_agg[_user_agg["first_event_ts"] <= _cutoff].copy()
_excluded = len(_user_agg) - len(cohort_users)

print(f"\n🎯 Cohort filter (first_event_ts ≤ {COHORT_CUTOFF.date()}):")
print(f"   Kept:     {len(cohort_users):,} users")
print(f"   Excluded: {_excluded:,} users")

# Filter events to cohort users only
_cohort_ids = set(cohort_users["user_id_canon"])
events = events_all[events_all["user_id_canon"].isin(_cohort_ids)].copy()

print(f"   Events after filter: {len(events):,} rows")
print(f"\n✅ cohort_users ({len(cohort_users):,} rows) and events ({len(events):,} rows) ready for downstream")
