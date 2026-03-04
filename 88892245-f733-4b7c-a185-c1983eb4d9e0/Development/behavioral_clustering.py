
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

_BG = "#1D1D20"
_TXT = "#fbfbff"
_TXT2 = "#909094"
_COLS = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
         "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
_HL = "#ffd400"

# ═════════════════════════════════════════════════════
# 0. PREPARE EARLY-WINDOW FEATURES FOR CLUSTERING
# ═════════════════════════════════════════════════════
_TARGETS = ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]
_META_COLS = ["user_id_canon", "split"] + _TARGETS
_FEATURE_COLS = [c for c in modeling_df.columns if c not in _META_COLS]

# Use only continuous early-window behavioral features (not one-hot encoded)
_CLUSTER_FEATURES = [c for c in _FEATURE_COLS 
                     if not c.startswith("feat_primary_") 
                     and c not in ["feat_signup_dow", "feat_signup_hour"]]

print(f"📊 Clustering on {len(_CLUSTER_FEATURES)} behavioral features")
print(f"   (excluded device/os/browser/country dummies and time-of-signup)")

_X_cluster = modeling_df[_CLUSTER_FEATURES].values.copy()

# Standardize
_scaler_cluster = StandardScaler()
_X_cluster_scaled = _scaler_cluster.fit_transform(_X_cluster)

# ═════════════════════════════════════════════════════
# 1. SILHOUETTE SCORE SEARCH (k=4 to 6)
# ═════════════════════════════════════════════════════
_k_range = range(4, 7)
_silhouette_scores = {}

for _k in _k_range:
    _km = KMeans(n_clusters=_k, random_state=42, n_init=10, max_iter=300)
    _labels = _km.fit_predict(_X_cluster_scaled)
    _sil = silhouette_score(_X_cluster_scaled, _labels, sample_size=min(1000, len(_X_cluster_scaled)))
    _silhouette_scores[_k] = _sil
    print(f"   k={_k}: silhouette = {_sil:.4f}")

# NOTE: renamed to behavioral_cluster_best_k to avoid conflict with kmeans_archetype_clustering
behavioral_cluster_best_k = max(_silhouette_scores, key=_silhouette_scores.get)
print(f"\n🏆 Best k = {behavioral_cluster_best_k} (silhouette = {_silhouette_scores[behavioral_cluster_best_k]:.4f})")

# ═════════════════════════════════════════════════════
# 2. FIT FINAL KMEANS WITH BEST K
# ═════════════════════════════════════════════════════
_km_final = KMeans(n_clusters=behavioral_cluster_best_k, random_state=42, n_init=20, max_iter=500)
_cluster_labels = _km_final.fit_predict(_X_cluster_scaled)

# Add cluster labels to modeling_df copy
_cluster_df = modeling_df.copy()
_cluster_df["cluster_id"] = _cluster_labels

# ═════════════════════════════════════════════════════
# 3. CHARACTERIZE CLUSTERS — centroid profiles
# ═════════════════════════════════════════════════════
_centroids_scaled = _km_final.cluster_centers_
_centroids_original = _scaler_cluster.inverse_transform(_centroids_scaled)
_centroid_df = pd.DataFrame(_centroids_original, columns=_CLUSTER_FEATURES)

# Key profile features
_PROFILE_FEATURES = [
    "feat_event_count", "feat_active_days", "feat_n_sessions",
    "feat_ratio_agent", "feat_ratio_block_ops", "feat_ratio_canvas",
    "feat_ratio_onboarding", "feat_mean_session_duration_min",
    "feat_distinct_events", "feat_distinct_canvases",
    "feat_early_deploy_count", "feat_collab_actions",
    "feat_onboarding_completed", "feat_tour_finished",
]

print("\n" + "=" * 80)
print("📊 CLUSTER CENTROID PROFILES")
print("=" * 80)

_profile_available = [f for f in _PROFILE_FEATURES if f in _CLUSTER_FEATURES]

for _feat in _profile_available:
    _vals = [f"{_centroid_df.loc[_c, _feat]:.2f}" for _c in range(behavioral_cluster_best_k)]
    _clean = _feat.replace("feat_", "").replace("_", " ").title()
    print(f"   {_clean:40s}  " + "  |  ".join([f"C{_c}={_vals[_c]:>8s}" for _c in range(behavioral_cluster_best_k)]))

# ═════════════════════════════════════════════════════
# 4. NAME CLUSTERS BASED ON CENTROID PROFILES — data-driven
# ═════════════════════════════════════════════════════
# Sort clusters by activity level (event_count) for consistent naming
_activity_rank = _centroid_df["feat_event_count"].values
_sorted_cluster_ids = np.argsort(_activity_rank)

cluster_names = {}
for _rank, _c in enumerate(_sorted_cluster_ids):
    _row = _centroid_df.loc[_c]
    _event_count = _row["feat_event_count"]
    _active_days = _row["feat_active_days"]
    _agent_ratio = _row.get("feat_ratio_agent", 0)
    _block_ratio = _row.get("feat_ratio_block_ops", 0)
    _canvas_ratio = _row.get("feat_ratio_canvas", 0)
    _onboard_ratio = _row.get("feat_ratio_onboarding", 0)
    _sessions = _row["feat_n_sessions"]
    _deploy = _row.get("feat_early_deploy_count", 0)
    _session_dur = _row.get("feat_mean_session_duration_min", 0)
    _size = (_cluster_labels == _c).sum()
    
    # Multi-criteria naming
    if _event_count < 8 and _active_days <= 1.1 and _agent_ratio > 0.5:
        _name = "Quick AI Trier"
    elif _event_count < 8 and _active_days <= 1.2 and _onboard_ratio > 0.3:
        _name = "Onboarding-Only Visitor"
    elif _event_count < 15 and _active_days <= 1.5:
        _name = "Casual Visitor"
    elif _agent_ratio > 0.5 and _event_count > 80:
        _name = "AI-First Power User"
    elif _block_ratio > 0.15 and _active_days >= 2:
        _name = "Hands-On Builder"
    elif _canvas_ratio > 0.3 and _sessions > 20:
        _name = "Canvas-Heavy Navigator"
    elif _deploy > 1:
        _name = "Deployer / Collaborator"
    elif _event_count > 50 and _session_dur > 5:
        _name = "Deep Explorer"
    elif _onboard_ratio > 0.2 and _event_count < 30:
        _name = "Onboarding Completer"
    else:
        _name = f"Moderate User (Tier {_rank+1})"
    
    cluster_names[_c] = _name
    print(f"\n🏷️  Cluster {_c}: {_name}  (n={_size}, {_size/len(_cluster_labels)*100:.1f}%)")
    print(f"     events={_event_count:.0f}, days={_active_days:.1f}, sessions={_sessions:.0f}, "
          f"agent%={_agent_ratio*100:.0f}%, block%={_block_ratio*100:.0f}%, dur={_session_dur:.1f}min")

_cluster_df["cluster_name"] = _cluster_df["cluster_id"].map(cluster_names)

# ═════════════════════════════════════════════════════
# 5. OUTCOME RATES BY CLUSTER
# ═════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("📊 RETENTION & UPGRADE RATES BY CLUSTER")
print("=" * 80)

cluster_outcome_table = _cluster_df.groupby("cluster_name").agg(
    n_users=("user_id_canon", "count"),
    ret_30d_rate=("y_ret_30d", "mean"),
    ret_90d_rate=("y_ret_90d", "mean"),
    upgrade_60d_rate=("y_upgrade_60d", "mean"),
).reset_index()

cluster_outcome_table = cluster_outcome_table.sort_values("ret_90d_rate", ascending=False)

print(f"\n{'Cluster Name':40s} {'N':>6s} {'Ret30d':>8s} {'Ret90d':>8s} {'Upg60d':>8s}")
print("-" * 75)
for _, _r in cluster_outcome_table.iterrows():
    print(f"{_r['cluster_name']:40s} {_r['n_users']:6.0f} "
          f"{_r['ret_30d_rate']*100:7.1f}% {_r['ret_90d_rate']*100:7.1f}% {_r['upgrade_60d_rate']*100:7.1f}%")

_overall_30 = _cluster_df["y_ret_30d"].mean()
_overall_90 = _cluster_df["y_ret_90d"].mean()
_overall_up = _cluster_df["y_upgrade_60d"].mean()
print(f"{'OVERALL':40s} {len(_cluster_df):6d} {_overall_30*100:7.1f}% {_overall_90*100:7.1f}% {_overall_up*100:7.1f}%")

# ═════════════════════════════════════════════════════
# 6. CLUSTER VISUALIZATION — outcome comparison bar chart
# ═════════════════════════════════════════════════════
_target_labels = {"ret_30d_rate": "30-Day Retention", "ret_90d_rate": "90-Day Retention", "upgrade_60d_rate": "60-Day Upgrade"}
_metrics = ["ret_30d_rate", "ret_90d_rate", "upgrade_60d_rate"]

fig_cluster_outcomes, _ax = plt.subplots(figsize=(14, 7), facecolor=_BG)
_ax.set_facecolor(_BG)

_x = np.arange(len(cluster_outcome_table))
_width = 0.25

for _m_idx, _metric in enumerate(_metrics):
    _vals = cluster_outcome_table[_metric].values * 100
    _offset = (_m_idx - 1) * _width
    _bars = _ax.bar(_x + _offset, _vals, _width, label=_target_labels[_metric],
                    color=_COLS[_m_idx], edgecolor="none", alpha=0.9)
    for _bar, _v in zip(_bars, _vals):
        if _v > 0.5:
            _ax.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height() + 0.3,
                     f"{_v:.1f}%", ha="center", va="bottom", color=_TXT, fontsize=8, fontweight="bold")

_ax.set_xticks(_x)
_ax.set_xticklabels(cluster_outcome_table["cluster_name"].values, fontsize=9, rotation=20, ha="right")
_ax.set_ylabel("Rate (%)", color=_TXT, fontsize=12)
_ax.set_title("Retention & Upgrade Rates by Behavioral Cluster", color=_TXT, fontsize=14, fontweight="bold", pad=15)
_ax.tick_params(colors=_TXT2, labelsize=10)
_ax.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10)
_ax.spines["top"].set_visible(False)
_ax.spines["right"].set_visible(False)
_ax.spines["bottom"].set_color(_TXT2)
_ax.spines["left"].set_color(_TXT2)

_ax.axhline(_overall_30*100, color=_COLS[0], linestyle="--", alpha=0.4, linewidth=1)
_ax.axhline(_overall_90*100, color=_COLS[1], linestyle="--", alpha=0.4, linewidth=1)
_ax.axhline(_overall_up*100, color=_COLS[2], linestyle="--", alpha=0.4, linewidth=1)

plt.tight_layout()
plt.show()

# ═════════════════════════════════════════════════════
# 7. CLUSTER SIZE DISTRIBUTION
# ═════════════════════════════════════════════════════
fig_cluster_sizes, _ax2 = plt.subplots(figsize=(10, 6), facecolor=_BG)
_ax2.set_facecolor(_BG)

_sizes = cluster_outcome_table["n_users"].values
_names_short = cluster_outcome_table["cluster_name"].values
_bar_colors = [_COLS[i % len(_COLS)] for i in range(len(_names_short))]

_bars = _ax2.barh(range(len(_names_short)), _sizes, color=_bar_colors, edgecolor="none", alpha=0.9)
for _bar, _s in zip(_bars, _sizes):
    _ax2.text(_bar.get_width() + 5, _bar.get_y() + _bar.get_height()/2,
              f"{_s} ({_s/len(_cluster_df)*100:.1f}%)", va="center", color=_TXT, fontsize=11, fontweight="bold")

_ax2.set_yticks(range(len(_names_short)))
_ax2.set_yticklabels(_names_short, fontsize=11, color=_TXT)
_ax2.set_xlabel("Number of Users", color=_TXT, fontsize=12)
_ax2.set_title("Behavioral Cluster Size Distribution", color=_TXT, fontsize=14, fontweight="bold", pad=15)
_ax2.tick_params(colors=_TXT2, labelsize=10)
_ax2.spines["top"].set_visible(False)
_ax2.spines["right"].set_visible(False)
_ax2.spines["bottom"].set_color(_TXT2)
_ax2.spines["left"].set_color(_TXT2)

plt.tight_layout()
plt.show()

print("\n✅ Behavioral clustering complete.")
