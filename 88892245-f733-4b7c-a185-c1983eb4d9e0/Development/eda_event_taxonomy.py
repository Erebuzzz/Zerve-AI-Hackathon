
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
# 1. Event Taxonomy — Top 30 event types
# ─────────────────────────────────────────────────────
_evt = events["event"].value_counts().head(30).sort_values()

fig_event_taxonomy, ax = plt.subplots(figsize=(12, 10), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

_bars = ax.barh(_evt.index, _evt.values, color=ZERVE_COLORS[0], edgecolor="none", height=0.7)

# Annotate bars with counts
for _bar, _val in zip(_bars, _evt.values):
    ax.text(_val + _evt.max() * 0.01, _bar.get_y() + _bar.get_height() / 2,
            f"{_val:,}", va="center", ha="left", fontsize=8, color=TEXT_SECONDARY)

ax.set_xlabel("Event Count", color=TEXT_PRIMARY, fontsize=12)
ax.set_title("Event Taxonomy — Top 30 Event Types (Cohort Users)", 
             color=TEXT_PRIMARY, fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(TEXT_SECONDARY)
ax.spines["left"].set_color(TEXT_SECONDARY)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────
# 2. Event Category Breakdown (grouped taxonomy)
# ─────────────────────────────────────────────────────
def _categorize_event(evt_name):
    if "agent" in evt_name: return "AI Agent"
    if "credit" in evt_name: return "Credits / Billing"
    if "block" in evt_name or "run_" in evt_name: return "Block Operations"
    if "canvas" in evt_name: return "Canvas"
    if "sign" in evt_name or "user" in evt_name or "onboarding" in evt_name: return "Auth & Onboarding"
    if "fullscreen" in evt_name: return "UI / Fullscreen"
    if "edge" in evt_name or "layer" in evt_name: return "DAG / Layers"
    if "file" in evt_name: return "Files"
    if "app" in evt_name or "scheduled" in evt_name: return "Apps & Jobs"
    return "Other"

_cat_counts = events["event"].map(_categorize_event).value_counts().sort_values()

fig_event_categories, ax2 = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
ax2.set_facecolor(BG_COLOR)

_colors = [ZERVE_COLORS[i % len(ZERVE_COLORS)] for i in range(len(_cat_counts))]
_bars2 = ax2.barh(_cat_counts.index, _cat_counts.values, color=_colors, edgecolor="none", height=0.6)

for _bar, _val in zip(_bars2, _cat_counts.values):
    ax2.text(_val + _cat_counts.max() * 0.01, _bar.get_y() + _bar.get_height() / 2,
             f"{_val:,}", va="center", ha="left", fontsize=10, color=TEXT_SECONDARY)

ax2.set_xlabel("Event Count", color=TEXT_PRIMARY, fontsize=12)
ax2.set_title("Event Categories — Grouped Taxonomy", 
              color=TEXT_PRIMARY, fontsize=14, fontweight="bold", pad=15)
ax2.tick_params(colors=TEXT_SECONDARY, labelsize=10)
ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_color(TEXT_SECONDARY)
ax2.spines["left"].set_color(TEXT_SECONDARY)

plt.tight_layout()
plt.show()

print("✅ Event taxonomy charts rendered")
