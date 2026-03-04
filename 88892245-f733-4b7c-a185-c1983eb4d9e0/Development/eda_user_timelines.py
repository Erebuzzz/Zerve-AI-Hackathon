
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# ─────────────────────────────────────────────────────
# Zerve Design System
# ─────────────────────────────────────────────────────
BG_COLOR = "#1D1D20"
TEXT_PRIMARY = "#fbfbff"
TEXT_SECONDARY = "#909094"
ZERVE_COLORS = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
                "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
HIGHLIGHT = "#ffd400"

# ─────────────────────────────────────────────────────
# 1. User Sign-Up Timeline (first_event_ts by week)
# ─────────────────────────────────────────────────────
_first_dates = cohort_users["first_event_ts"].dt.date
_weekly = pd.Series(_first_dates.values).value_counts().sort_index()
_weekly.index = pd.to_datetime(_weekly.index)
_weekly_resampled = _weekly.resample("W").sum()

fig_signup_timeline, ax1 = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
ax1.set_facecolor(BG_COLOR)
ax1.fill_between(_weekly_resampled.index, _weekly_resampled.values, alpha=0.3, color=ZERVE_COLORS[0])
ax1.plot(_weekly_resampled.index, _weekly_resampled.values, color=ZERVE_COLORS[0], linewidth=2)
ax1.set_title("New User Arrivals by Week (First Event Date)", 
              color=TEXT_PRIMARY, fontsize=14, fontweight="bold", pad=15)
ax1.set_ylabel("New Users", color=TEXT_PRIMARY, fontsize=12)
ax1.set_xlabel("Week", color=TEXT_PRIMARY, fontsize=12)
ax1.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_color(TEXT_SECONDARY)
ax1.spines["left"].set_color(TEXT_SECONDARY)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────
# 2. Active Days Distribution
# ─────────────────────────────────────────────────────
fig_active_days, ax2 = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
ax2.set_facecolor(BG_COLOR)

_days = cohort_users["distinct_days"].clip(upper=50)
ax2.hist(_days, bins=50, color=ZERVE_COLORS[1], edgecolor=BG_COLOR, alpha=0.85)
ax2.axvline(_days.median(), color=HIGHLIGHT, linestyle="--", linewidth=2, label=f"Median: {cohort_users['distinct_days'].median():.0f} days")
ax2.set_title("Distribution of Active Days per User", 
              color=TEXT_PRIMARY, fontsize=14, fontweight="bold", pad=15)
ax2.set_xlabel("Distinct Active Days (capped at 50)", color=TEXT_PRIMARY, fontsize=12)
ax2.set_ylabel("Number of Users", color=TEXT_PRIMARY, fontsize=12)
ax2.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax2.legend(facecolor=BG_COLOR, edgecolor=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY, fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_color(TEXT_SECONDARY)
ax2.spines["left"].set_color(TEXT_SECONDARY)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────
# 3. Total Events per User (log scale)
# ─────────────────────────────────────────────────────
fig_events_per_user, ax3 = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
ax3.set_facecolor(BG_COLOR)

_log_events = np.log10(cohort_users["total_events"].clip(lower=1))
ax3.hist(_log_events, bins=60, color=ZERVE_COLORS[2], edgecolor=BG_COLOR, alpha=0.85)
_med = np.log10(cohort_users["total_events"].median())
ax3.axvline(_med, color=HIGHLIGHT, linestyle="--", linewidth=2, 
            label=f"Median: {cohort_users['total_events'].median():.0f} events")
ax3.set_title("Distribution of Total Events per User (log₁₀ scale)", 
              color=TEXT_PRIMARY, fontsize=14, fontweight="bold", pad=15)
ax3.set_xlabel("log₁₀(Total Events)", color=TEXT_PRIMARY, fontsize=12)
ax3.set_ylabel("Number of Users", color=TEXT_PRIMARY, fontsize=12)
ax3.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax3.legend(facecolor=BG_COLOR, edgecolor=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY, fontsize=11)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["bottom"].set_color(TEXT_SECONDARY)
ax3.spines["left"].set_color(TEXT_SECONDARY)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────
# 4. Sessions vs Active Days scatter
# ─────────────────────────────────────────────────────
fig_sessions_vs_days, ax4 = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
ax4.set_facecolor(BG_COLOR)

ax4.scatter(cohort_users["distinct_days"], cohort_users["n_sessions"],
            alpha=0.3, s=15, color=ZERVE_COLORS[4], edgecolors="none")
ax4.set_title("Sessions vs Active Days per User", 
              color=TEXT_PRIMARY, fontsize=14, fontweight="bold", pad=15)
ax4.set_xlabel("Distinct Active Days", color=TEXT_PRIMARY, fontsize=12)
ax4.set_ylabel("Number of Sessions", color=TEXT_PRIMARY, fontsize=12)
ax4.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["bottom"].set_color(TEXT_SECONDARY)
ax4.spines["left"].set_color(TEXT_SECONDARY)
plt.tight_layout()
plt.show()

print("✅ User timeline pattern charts rendered")
