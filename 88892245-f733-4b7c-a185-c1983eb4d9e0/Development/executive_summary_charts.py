
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════
# ZERVE DESIGN SYSTEM
# ═════════════════════════════════════════════════════
_BG = "#1D1D20"
_TXT = "#fbfbff"
_TXT2 = "#909094"
_COLS = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
         "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
_HL = "#ffd400"
_SUCCESS = "#17b26a"
_WARN = "#f04438"

_TARGETS = ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]
_META_COLS = ["user_id_canon", "split"] + _TARGETS
_FEATURE_COLS = [c for c in modeling_df.columns if c not in _META_COLS]

def _clean_name(s):
    return s.replace("feat_", "").replace("_", " ").title()

# ═══════════════════════════════════════════════════
# 1. TOP 5 DRIVERS OF SUCCESS — Horizontal Bar Chart
# ═══════════════════════════════════════════════════
# Use SHAP importance for primary objective (y_upgrade_60d)
_primary_imp = shap_importance_tables[primary_objective].head(5)

fig_top5_drivers, _ax = plt.subplots(figsize=(14, 6), facecolor=_BG)
_ax.set_facecolor(_BG)

_names = _primary_imp["Clean Name"].values[::-1]
_values = _primary_imp["Mean |SHAP|"].values[::-1]
_bar_colors = [_HL if i == len(_names)-1 else _COLS[i % len(_COLS)] for i in range(len(_names))][::-1]

_bars = _ax.barh(range(len(_names)), _values, color=_bar_colors, edgecolor="none", alpha=0.92, height=0.6)

for _bar, _v, _n in zip(_bars, _values, _names):
    _ax.text(_bar.get_width() + 0.02, _bar.get_y() + _bar.get_height()/2,
             f"{_v:.3f}", va="center", ha="left", color=_TXT, fontsize=12, fontweight="bold")

_ax.set_yticks(range(len(_names)))
_ax.set_yticklabels(_names, fontsize=13, color=_TXT, fontweight="bold")
_ax.set_xlabel("Mean |SHAP| Value", color=_TXT, fontsize=12)

_primary_lbl = {"y_ret_30d": "30-Day Retention", "y_ret_90d": "90-Day Retention", "y_upgrade_60d": "60-Day Upgrade"}[primary_objective]
_ax.set_title(f"Top 5 Drivers of {_primary_lbl} (Primary Objective)",
              color=_HL, fontsize=16, fontweight="bold", pad=18)
_ax.tick_params(colors=_TXT2, labelsize=11)
_ax.spines["top"].set_visible(False)
_ax.spines["right"].set_visible(False)
_ax.spines["bottom"].set_color(_TXT2)
_ax.spines["left"].set_color(_TXT2)

# Add rank numbers
for _i, _n in enumerate(_names):
    _rank = len(_names) - _i
    _ax.text(-0.015 * _values.max(), _i, f"#{_rank}", va="center", ha="right",
             color=_HL if _rank == 1 else _TXT2, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════
# 2. LIFT CHART — Primary Objective (GBT Model)
# ═══════════════════════════════════════════════════
_test_mask = modeling_df["split"] == "test"
_y_test_primary = modeling_df.loc[_test_mask, primary_objective].values
_y_prob_gbt = all_predictions[(primary_objective, "GBT")]
_y_prob_lr = all_predictions[(primary_objective, "L2_LR")]
_base_rate = _y_test_primary.mean()

_percentiles = np.arange(0.01, 1.01, 0.01)
_lift_gbt = []
_lift_lr = []
_capture_gbt = []

for _pct in _percentiles:
    _k = max(1, int(len(_y_test_primary) * _pct))
    
    _top_gbt = np.argsort(_y_prob_gbt)[::-1][:_k]
    _rate_gbt = _y_test_primary[_top_gbt].mean()
    _lift_gbt.append(_rate_gbt / _base_rate if _base_rate > 0 else 1.0)
    _capture_gbt.append(_y_test_primary[_top_gbt].sum() / _y_test_primary.sum() if _y_test_primary.sum() > 0 else 0)
    
    _top_lr = np.argsort(_y_prob_lr)[::-1][:_k]
    _rate_lr = _y_test_primary[_top_lr].mean()
    _lift_lr.append(_rate_lr / _base_rate if _base_rate > 0 else 1.0)

fig_lift_chart, _axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=_BG)

# Lift curve
_ax1 = _axes[0]
_ax1.set_facecolor(_BG)
_ax1.plot(_percentiles * 100, _lift_gbt, color=_COLS[0], linewidth=2.5, label="Gradient-Boosted Trees", zorder=3)
_ax1.plot(_percentiles * 100, _lift_lr, color=_COLS[1], linewidth=2, label="L2 Logistic Regression", alpha=0.7, zorder=2)
_ax1.axhline(1, color=_TXT2, linestyle="--", linewidth=1, alpha=0.5, label="Random (Lift=1)")
_ax1.fill_between(_percentiles * 100, 1, _lift_gbt, alpha=0.1, color=_COLS[0])
_ax1.set_xlabel("Top % of Users (by predicted probability)", color=_TXT, fontsize=12)
_ax1.set_ylabel("Lift (vs. Base Rate)", color=_TXT, fontsize=12)
_ax1.set_title(f"Lift Chart — {_primary_lbl}", color=_TXT, fontsize=14, fontweight="bold", pad=12)
_ax1.tick_params(colors=_TXT2, labelsize=10)
_ax1.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10, loc="upper right")
_ax1.spines["top"].set_visible(False)
_ax1.spines["right"].set_visible(False)
_ax1.spines["bottom"].set_color(_TXT2)
_ax1.spines["left"].set_color(_TXT2)

# Annotate key points
_lift_at_10 = _lift_gbt[9]
_lift_at_20 = _lift_gbt[19]
_ax1.annotate(f"Top 10%: {_lift_at_10:.1f}x lift", xy=(10, _lift_at_10),
              xytext=(25, _lift_at_10 + 0.5), fontsize=10, color=_HL, fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=_HL, lw=1.5))
_ax1.annotate(f"Top 20%: {_lift_at_20:.1f}x lift", xy=(20, _lift_at_20),
              xytext=(35, _lift_at_20 + 0.3), fontsize=10, color=_COLS[2], fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=_COLS[2], lw=1.5))

# Cumulative gains curve
_ax2 = _axes[1]
_ax2.set_facecolor(_BG)
_ax2.plot(_percentiles * 100, np.array(_capture_gbt) * 100, color=_COLS[0], linewidth=2.5, label="GBT Model")
_ax2.plot([0, 100], [0, 100], color=_TXT2, linestyle="--", linewidth=1, alpha=0.5, label="Random")
_ax2.fill_between(_percentiles * 100, _percentiles * 100, np.array(_capture_gbt) * 100, alpha=0.1, color=_COLS[0])
_ax2.set_xlabel("Top % of Users Targeted", color=_TXT, fontsize=12)
_ax2.set_ylabel("% of Positive Outcomes Captured", color=_TXT, fontsize=12)
_ax2.set_title(f"Cumulative Gains — {_primary_lbl}", color=_TXT, fontsize=14, fontweight="bold", pad=12)
_ax2.tick_params(colors=_TXT2, labelsize=10)
_ax2.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10)
_ax2.spines["top"].set_visible(False)
_ax2.spines["right"].set_visible(False)
_ax2.spines["bottom"].set_color(_TXT2)
_ax2.spines["left"].set_color(_TXT2)

_cap_at_20 = _capture_gbt[19] * 100
_ax2.annotate(f"Top 20% captures {_cap_at_20:.0f}%\nof all upgrades",
              xy=(20, _cap_at_20), xytext=(35, _cap_at_20 + 10), fontsize=10,
              color=_HL, fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=_HL, lw=1.5))

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════
# 3. ARCHETYPE OUTCOME TABLE (styled console output)
# ═══════════════════════════════════════════════════
print("=" * 90)
print("📊 BEHAVIORAL ARCHETYPES & OUTCOME RATES")
print("=" * 90)

_overall_30 = modeling_df["y_ret_30d"].mean()
_overall_90 = modeling_df["y_ret_90d"].mean()
_overall_up = modeling_df["y_upgrade_60d"].mean()

print(f"\n{'Archetype':<28s} {'Users':>7s} {'Share':>7s} │ {'Ret30d':>8s} {'Ret90d':>8s} {'Upg60d':>8s} │ {'Risk Level':<15s}")
print("─" * 100)

_risk_map = {
    "Hands-On Builder": "🟢 Low",
    "AI-First Power User": "🟡 Medium",
    "Onboarding-Only Visitor": "🟠 High",
    "Casual Visitor": "🔴 Critical"
}

for _, _r in cluster_outcome_table.iterrows():
    _n = _r["n_users"]
    _share = _n / len(modeling_df) * 100
    _r30 = _r["ret_30d_rate"] * 100
    _r90 = _r["ret_90d_rate"] * 100
    _up = _r["upgrade_60d_rate"] * 100
    _risk = _risk_map.get(_r["cluster_name"], "⚪ Unknown")
    print(f"{_r['cluster_name']:<28s} {_n:>7d} {_share:>6.1f}% │ {_r30:>7.1f}% {_r90:>7.1f}% {_up:>7.1f}% │ {_risk:<15s}")

print("─" * 100)
print(f"{'OVERALL':<28s} {len(modeling_df):>7d} {100.0:>6.1f}% │ {_overall_30*100:>7.1f}% {_overall_90*100:>7.1f}% {_overall_up*100:>7.1f}% │")
print()

# ═══════════════════════════════════════════════════
# 4. ONE-SLIDE EXECUTIVE VIEW
# ═══════════════════════════════════════════════════
fig_executive_slide, _gs_axes = plt.subplots(2, 3, figsize=(22, 14), facecolor=_BG,
                                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})

# Title banner
fig_executive_slide.text(0.5, 0.98, "ZERVE USER ACTIVATION & RETENTION — EXECUTIVE VIEW",
                          ha="center", va="top", fontsize=20, fontweight="bold", color=_HL,
                          transform=fig_executive_slide.transFigure)
fig_executive_slide.text(0.5, 0.955, f"Cohort: 1,472 users  |  Early Window: 7 days  |  Primary Objective: {_primary_lbl}",
                          ha="center", va="top", fontsize=12, color=_TXT2,
                          transform=fig_executive_slide.transFigure)

# ─── Panel 1: Top 5 Drivers (bar) ───
_ax_d = _gs_axes[0, 0]
_ax_d.set_facecolor(_BG)
_top5 = shap_importance_tables[primary_objective].head(5)
_y_pos = range(5)
_vals_d = _top5["Mean |SHAP|"].values[::-1]
_names_d = _top5["Clean Name"].values[::-1]
_ax_d.barh(_y_pos, _vals_d, color=[_HL if i==4 else _COLS[i%len(_COLS)] for i in range(5)][::-1],
           height=0.55, alpha=0.9, edgecolor="none")
_ax_d.set_yticks(_y_pos)
_ax_d.set_yticklabels(_names_d, fontsize=9, color=_TXT)
_ax_d.set_title("Top 5 Drivers", color=_TXT, fontsize=13, fontweight="bold", pad=10)
_ax_d.tick_params(colors=_TXT2, labelsize=8)
_ax_d.spines["top"].set_visible(False)
_ax_d.spines["right"].set_visible(False)
_ax_d.spines["bottom"].set_color(_TXT2)
_ax_d.spines["left"].set_color(_TXT2)

# ─── Panel 2: Lift chart (line) ───
_ax_l = _gs_axes[0, 1]
_ax_l.set_facecolor(_BG)
_ax_l.plot(_percentiles * 100, _lift_gbt, color=_COLS[0], linewidth=2)
_ax_l.axhline(1, color=_TXT2, linestyle="--", linewidth=0.8, alpha=0.5)
_ax_l.fill_between(_percentiles * 100, 1, _lift_gbt, alpha=0.1, color=_COLS[0])
_ax_l.set_xlabel("Top % Targeted", color=_TXT2, fontsize=9)
_ax_l.set_ylabel("Lift", color=_TXT2, fontsize=9)
_ax_l.set_title(f"Lift Chart — {_primary_lbl}", color=_TXT, fontsize=13, fontweight="bold", pad=10)
_ax_l.tick_params(colors=_TXT2, labelsize=8)
_ax_l.spines["top"].set_visible(False)
_ax_l.spines["right"].set_visible(False)
_ax_l.spines["bottom"].set_color(_TXT2)
_ax_l.spines["left"].set_color(_TXT2)
_ax_l.text(10, _lift_at_10, f"{_lift_at_10:.1f}x", color=_HL, fontsize=10, fontweight="bold")

# ─── Panel 3: Model Performance KPIs ───
_ax_k = _gs_axes[0, 2]
_ax_k.set_facecolor(_BG)
_ax_k.axis("off")
_ax_k.set_title("Model Performance", color=_TXT, fontsize=13, fontweight="bold", pad=10)

_kpi_data = [
    ("GBT ROC-AUC", f"{all_results[primary_objective][3]['ROC-AUC']:.3f}"),
    ("GBT PR-AUC", f"{all_results[primary_objective][3]['PR-AUC']:.3f}"),
    (f"Lift @ Top 10%", f"{all_results[primary_objective][3]['Lift@10%']:.1f}x"),
    (f"Lift @ Top 20%", f"{all_results[primary_objective][3]['Lift@20%']:.1f}x"),
    ("Base Rate", f"{_base_rate*100:.1f}%"),
    ("Cohort Size", f"1,472 users"),
]
for _i, (_label, _val) in enumerate(_kpi_data):
    _ax_k.text(0.05, 0.88 - _i*0.16, _label, transform=_ax_k.transAxes,
               fontsize=11, color=_TXT2, va="center")
    _ax_k.text(0.95, 0.88 - _i*0.16, _val, transform=_ax_k.transAxes,
               fontsize=13, color=_HL, fontweight="bold", va="center", ha="right")

# ─── Panel 4: Archetype Outcome Bars ───
_ax_a = _gs_axes[1, 0]
_ax_a.set_facecolor(_BG)
_x = np.arange(len(cluster_outcome_table))
_width = 0.25
_metrics_list = ["ret_30d_rate", "ret_90d_rate", "upgrade_60d_rate"]
_metric_labels = ["Ret 30d", "Ret 90d", "Upgrade"]
for _mi, _m in enumerate(_metrics_list):
    _vals_a = cluster_outcome_table[_m].values * 100
    _ax_a.bar(_x + (_mi-1)*_width, _vals_a, _width, color=_COLS[_mi], alpha=0.9, edgecolor="none")
_ax_a.set_xticks(_x)
_ax_a.set_xticklabels([n[:12] + "…" if len(n) > 12 else n for n in cluster_outcome_table["cluster_name"].values],
                       fontsize=8, rotation=25, ha="right", color=_TXT)
_ax_a.set_ylabel("Rate %", color=_TXT2, fontsize=9)
_ax_a.set_title("Archetype Outcomes", color=_TXT, fontsize=13, fontweight="bold", pad=10)
_ax_a.tick_params(colors=_TXT2, labelsize=8)
_ax_a.spines["top"].set_visible(False)
_ax_a.spines["right"].set_visible(False)
_ax_a.spines["bottom"].set_color(_TXT2)
_ax_a.spines["left"].set_color(_TXT2)
_legend_patches = [mpatches.Patch(color=_COLS[i], label=_metric_labels[i]) for i in range(3)]
_ax_a.legend(handles=_legend_patches, facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=8, loc="upper right")

# ─── Panel 5: Archetype Size Pie ───
_ax_p = _gs_axes[1, 1]
_ax_p.set_facecolor(_BG)
_sizes = cluster_outcome_table["n_users"].values
_labels_p = cluster_outcome_table["cluster_name"].values
_pie_colors = [_COLS[i % len(_COLS)] for i in range(len(_sizes))]
_wedges, _texts, _autotexts = _ax_p.pie(
    _sizes, labels=None, autopct="%1.0f%%", colors=_pie_colors,
    startangle=90, pctdistance=0.75, textprops={"fontsize": 9, "color": _TXT}
)
for _at in _autotexts:
    _at.set_fontweight("bold")
_ax_p.set_title("User Distribution", color=_TXT, fontsize=13, fontweight="bold", pad=10)
_legend_pie = _ax_p.legend(_labels_p, loc="lower center", fontsize=7,
                            facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT,
                            bbox_to_anchor=(0.5, -0.15), ncol=2)

# ─── Panel 6: Key Recommendations ───
_ax_r = _gs_axes[1, 2]
_ax_r.set_facecolor(_BG)
_ax_r.axis("off")
_ax_r.set_title("Key Recommendations", color=_TXT, fontsize=13, fontweight="bold", pad=10)

_recs = [
    ("1. Onboarding → Build", "Guide completers to\ncreate & run first block"),
    ("2. AI → Execution", "Prompt AI-only users\nto execute generated code"),
    ("3. Activation Checklist", "3-session, 2-day, 1-block\nmilestone nudges"),
    ("4. Lifecycle Campaigns", "Day-1, Day-3, Day-7\nre-engagement emails"),
]
for _i, (_title, _desc) in enumerate(_recs):
    _y_start = 0.92 - _i * 0.24
    _ax_r.text(0.05, _y_start, _title, transform=_ax_r.transAxes,
               fontsize=11, color=_HL, fontweight="bold", va="top")
    _ax_r.text(0.05, _y_start - 0.06, _desc, transform=_ax_r.transAxes,
               fontsize=9, color=_TXT2, va="top")

plt.show()

print("\n✅ Executive summary charts rendered successfully.")
print(f"   Primary Objective: {_primary_lbl}")
print(f"   Top Driver: {_primary_imp.iloc[0]['Clean Name']} (SHAP={_primary_imp.iloc[0]['Mean |SHAP|']:.3f})")
print(f"   GBT Lift@10%: {_lift_at_10:.1f}x  |  Lift@20%: {_lift_at_20:.1f}x")
print(f"   4 Archetypes identified with outcome rates from {cluster_outcome_table['ret_90d_rate'].min()*100:.1f}% to {cluster_outcome_table['ret_90d_rate'].max()*100:.1f}%")
