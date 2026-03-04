
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────────────
# Zerve Design System
# ─────────────────────────────────────────────────────
_BG = "#1D1D20"
_TXT = "#fbfbff"
_TXT2 = "#909094"
_COLS = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
         "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
_HL = "#ffd400"

# ─────────────────────────────────────────────────────
# Derive early-window behaviour flags (first 7 days)
# ─────────────────────────────────────────────────────
# Merge first_event_ts onto events
_ev = events.merge(
    cohort_users[["user_id_canon", "first_event_ts"]],
    on="user_id_canon", how="inner"
)
_ev["days_since_first"] = (_ev["timestamp"] - _ev["first_event_ts"]).dt.total_seconds() / 86400
_early = _ev[_ev["days_since_first"] <= EARLY_WINDOW_DAYS]

# Key early behaviours
_early_agent = set(_early[_early["event"].str.contains("agent", na=False)]["user_id_canon"])
_early_run   = set(_early[_early["event"] == "run_block"]["user_id_canon"])
_early_canvas = set(_early[_early["event"] == "canvas_create"]["user_id_canon"])
_early_credits = set(_early[_early["event"] == "credits_used"]["user_id_canon"])

# Build user-level flags
_cu = cohort_users.copy()
_cu["early_agent"]  = _cu["user_id_canon"].isin(_early_agent)
_cu["early_run"]    = _cu["user_id_canon"].isin(_early_run)
_cu["early_canvas"] = _cu["user_id_canon"].isin(_early_canvas)
_cu["early_credits"] = _cu["user_id_canon"].isin(_early_credits)

print("Early behaviour flags (first 7 days):")
for _flag in ["early_agent", "early_run", "early_canvas", "early_credits"]:
    _n = _cu[_flag].sum()
    print(f"  {_flag}: {_n:,} users ({_n/len(_cu)*100:.1f}%)")

# ─────────────────────────────────────────────────────
# Retention curves by early behaviour
# Week-over-week retention: % of users active in week N after first event
# ─────────────────────────────────────────────────────
_ev["week_num"] = (_ev["days_since_first"] // 7).astype(int)
_max_week = 8  # show 8 weeks

_weekly_active = _ev[_ev["week_num"] <= _max_week].groupby(
    ["user_id_canon", "week_num"]
).size().reset_index(name="_cnt")

# Merge flags
_weekly_active = _weekly_active.merge(
    _cu[["user_id_canon", "early_agent", "early_run", "early_canvas", "early_credits"]],
    on="user_id_canon", how="left"
)

_behaviors = {
    "Used AI Agent (early)": "early_agent",
    "Ran Blocks (early)": "early_run",
    "Created Canvas (early)": "early_canvas",
    "Used Credits (early)": "early_credits",
}

fig_retention_curves, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=_BG)
axes = axes.flatten()

for _idx, (_label, _col) in enumerate(_behaviors.items()):
    _ax = axes[_idx]
    _ax.set_facecolor(_BG)
    
    for _did, _color, _ls in [(True, _COLS[2], "-"), (False, _COLS[3], "--")]:
        _subset_users = set(_cu[_cu[_col] == _did]["user_id_canon"])
        _total = len(_subset_users)
        if _total == 0:
            continue
        _active_per_week = _weekly_active[
            _weekly_active["user_id_canon"].isin(_subset_users)
        ].groupby("week_num")["user_id_canon"].nunique()
        
        _weeks = range(0, _max_week + 1)
        _rates = [_active_per_week.get(_w, 0) / _total * 100 for _w in _weeks]
        
        _lbl = f"{'Yes' if _did else 'No'} (n={_total:,})"
        _ax.plot(_weeks, _rates, color=_color, linestyle=_ls, linewidth=2, 
                 marker="o", markersize=4, label=_lbl)
    
    _ax.set_title(_label, color=_TXT, fontsize=12, fontweight="bold")
    _ax.set_xlabel("Week After First Event", color=_TXT, fontsize=10)
    _ax.set_ylabel("% Active", color=_TXT, fontsize=10)
    _ax.tick_params(colors=_TXT2, labelsize=9)
    _ax.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=9)
    _ax.set_ylim(0, 105)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["bottom"].set_color(_TXT2)
    _ax.spines["left"].set_color(_TXT2)

plt.suptitle("Retention Curves by Key Early Behaviors (First 7 Days)", 
             color=_TXT, fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────
# Upgrade / Retention Rates by Activity Bins
# ─────────────────────────────────────────────────────
# Define "retained" = active in week 4+ (day 28+)
_retained_users = set(_ev[_ev["days_since_first"] >= 28]["user_id_canon"].unique())
_cu["retained_4w"] = _cu["user_id_canon"].isin(_retained_users)

# Define "upgraded" = user who used credits (proxy for paid/power user)
_upgrade_users = set(events[events["event"] == "credits_used"]["user_id_canon"])
_cu["upgraded"] = _cu["user_id_canon"].isin(_upgrade_users)

# Activity bins based on total events
_cu["activity_bin"] = pd.cut(
    _cu["total_events"],
    bins=[0, 5, 20, 50, 100, 500, float("inf")],
    labels=["1-5", "6-20", "21-50", "51-100", "101-500", "500+"]
)

_bin_stats = _cu.groupby("activity_bin", observed=True).agg(
    n_users=("user_id_canon", "size"),
    retention_rate=("retained_4w", "mean"),
    upgrade_rate=("upgraded", "mean"),
).reset_index()

print("\n📊 Retention & Upgrade Rates by Activity Bin:")
print(_bin_stats.to_string(index=False))

# Grouped bar chart
fig_rates_by_activity, ax_rates = plt.subplots(figsize=(12, 6), facecolor=_BG)
ax_rates.set_facecolor(_BG)

_x = np.arange(len(_bin_stats))
_w = 0.35

_bars1 = ax_rates.bar(_x - _w/2, _bin_stats["retention_rate"] * 100, _w, 
                       label="4-Week Retention %", color=_COLS[0], edgecolor="none")
_bars2 = ax_rates.bar(_x + _w/2, _bin_stats["upgrade_rate"] * 100, _w, 
                       label="Credit Usage (Upgrade) %", color=_COLS[1], edgecolor="none")

# Annotate with user counts
for _i, _row in _bin_stats.iterrows():
    ax_rates.text(_i, max(_row["retention_rate"], _row["upgrade_rate"]) * 100 + 2,
                  f"n={_row['n_users']:,}", ha="center", fontsize=9, color=_TXT2)

ax_rates.set_xticks(_x)
ax_rates.set_xticklabels(_bin_stats["activity_bin"], fontsize=11)
ax_rates.set_xlabel("Activity Bin (Total Events)", color=_TXT, fontsize=12)
ax_rates.set_ylabel("Rate (%)", color=_TXT, fontsize=12)
ax_rates.set_title("Retention & Upgrade Rates by User Activity Level", 
                    color=_TXT, fontsize=14, fontweight="bold", pad=15)
ax_rates.tick_params(colors=_TXT2, labelsize=10)
ax_rates.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=11)
ax_rates.set_ylim(0, 110)
ax_rates.spines["top"].set_visible(False)
ax_rates.spines["right"].set_visible(False)
ax_rates.spines["bottom"].set_color(_TXT2)
ax_rates.spines["left"].set_color(_TXT2)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────
# Retention by session count bins
# ─────────────────────────────────────────────────────
_cu["session_bin"] = pd.cut(
    _cu["n_sessions"],
    bins=[0, 1, 3, 5, 10, 20, float("inf")],
    labels=["1", "2-3", "4-5", "6-10", "11-20", "20+"]
)

_sess_stats = _cu.groupby("session_bin", observed=True).agg(
    n_users=("user_id_canon", "size"),
    retention_rate=("retained_4w", "mean"),
    upgrade_rate=("upgraded", "mean"),
).reset_index()

fig_rates_by_sessions, ax_sess = plt.subplots(figsize=(12, 6), facecolor=_BG)
ax_sess.set_facecolor(_BG)

_x2 = np.arange(len(_sess_stats))
_bars3 = ax_sess.bar(_x2 - _w/2, _sess_stats["retention_rate"] * 100, _w,
                      label="4-Week Retention %", color=_COLS[0], edgecolor="none")
_bars4 = ax_sess.bar(_x2 + _w/2, _sess_stats["upgrade_rate"] * 100, _w,
                      label="Credit Usage (Upgrade) %", color=_COLS[1], edgecolor="none")

for _i, _row in _sess_stats.iterrows():
    ax_sess.text(_i, max(_row["retention_rate"], _row["upgrade_rate"]) * 100 + 2,
                 f"n={_row['n_users']:,}", ha="center", fontsize=9, color=_TXT2)

ax_sess.set_xticks(_x2)
ax_sess.set_xticklabels(_sess_stats["session_bin"], fontsize=11)
ax_sess.set_xlabel("Session Count Bin", color=_TXT, fontsize=12)
ax_sess.set_ylabel("Rate (%)", color=_TXT, fontsize=12)
ax_sess.set_title("Retention & Upgrade Rates by Session Count", 
                   color=_TXT, fontsize=14, fontweight="bold", pad=15)
ax_sess.tick_params(colors=_TXT2, labelsize=10)
ax_sess.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=11)
ax_sess.set_ylim(0, 110)
ax_sess.spines["top"].set_visible(False)
ax_sess.spines["right"].set_visible(False)
ax_sess.spines["bottom"].set_color(_TXT2)
ax_sess.spines["left"].set_color(_TXT2)

plt.tight_layout()
plt.show()

print("\n✅ Retention curves + activity-binned rate charts rendered")
