
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

_BG    = "#1D1D20"
_TXT   = "#fbfbff"
_TXT2  = "#909094"
_COLS  = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
          "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
_HL    = "#ffd400"

# ═════════════════════════════════════════════════════════════════
# 0. BUILD CLUSTERING FEATURE MATRIX — exact 5 features from ticket
# ═════════════════════════════════════════════════════════════════
# Ticket: session, execution_ratio, agent_ratio, onboarding_ratio, active_days
# Map to modeling_df columns:
#   session           → feat_n_sessions
#   execution_ratio   → feat_ratio_block_ops  (block run/create ops)
#   agent_ratio       → feat_ratio_agent
#   onboarding_ratio  → feat_ratio_onboarding
#   active_days       → feat_active_days

_CLUSTER_COLS = [
    "feat_n_sessions",
    "feat_ratio_block_ops",   # execution_ratio
    "feat_ratio_agent",       # agent_ratio
    "feat_ratio_onboarding",  # onboarding_ratio
    "feat_active_days",
]

_df = modeling_df[["user_id_canon", "y_ret_30d", "y_ret_90d", "y_upgrade_60d"] + _CLUSTER_COLS].copy()

print("📊 Clustering on 5 specified behavioral features:")
for _c in _CLUSTER_COLS:
    print(f"   • {_c}: mean={_df[_c].mean():.3f}, std={_df[_c].std():.3f}")
print(f"   Total users: {len(_df):,}")

_X_raw = _df[_CLUSTER_COLS].values.copy()
_scaler = StandardScaler()
_X_scaled = _scaler.fit_transform(_X_raw)

# ═════════════════════════════════════════════════════════════════
# 1. SILHOUETTE SCORES FOR k=4,5,6  (stability with multiple seeds)
# ═════════════════════════════════════════════════════════════════
_K_VALUES = [4, 5, 6]
_N_SEEDS  = 5   # run each k with 5 seeds to check stability
_sil_matrix    = {}   # k → list of silhouette scores across seeds
_label_matrix  = {}   # k → list of label arrays across seeds

print("\n" + "="*65)
print("📐 SILHOUETTE SCORE + STABILITY ANALYSIS")
print("="*65)

for _k in _K_VALUES:
    _sil_scores  = []
    _label_arrays = []
    for _seed in range(42, 42 + _N_SEEDS):
        _km = KMeans(n_clusters=_k, random_state=_seed, n_init=15, max_iter=500)
        _lbl = _km.fit_predict(_X_scaled)
        _sil = silhouette_score(_X_scaled, _lbl)
        _sil_scores.append(_sil)
        _label_arrays.append(_lbl)
    _sil_matrix[_k]   = _sil_scores
    _label_matrix[_k] = _label_arrays
    print(f"   k={_k}: silhouette = {np.mean(_sil_scores):.4f} ± {np.std(_sil_scores):.4f} "
          f"  (min={np.min(_sil_scores):.4f}, max={np.max(_sil_scores):.4f})")

# ═════════════════════════════════════════════════════════════════
# 2. CLUSTER ASSIGNMENT CONSISTENCY  (Adjusted Rand Index across seeds)
# ═════════════════════════════════════════════════════════════════
from sklearn.metrics import adjusted_rand_score

print("\n" + "="*65)
print("🔁 CLUSTER ASSIGNMENT CONSISTENCY (ARI across seed pairs)")
print("="*65)

_ari_scores = {}
for _k in _K_VALUES:
    _lbls = _label_matrix[_k]
    _pairs = [(i, j) for i in range(len(_lbls)) for j in range(i+1, len(_lbls))]
    _aris  = [adjusted_rand_score(_lbls[a], _lbls[b]) for a, b in _pairs]
    _ari_scores[_k] = _aris
    print(f"   k={_k}: ARI = {np.mean(_aris):.4f} ± {np.std(_aris):.4f}  "
          f"(1.0 = perfectly consistent across seeds)")

# ═════════════════════════════════════════════════════════════════
# 3. SELECT BEST k — combine silhouette (quality) + ARI (stability)
# ═════════════════════════════════════════════════════════════════
# Composite score: 0.6 * norm_silhouette + 0.4 * norm_ari
_mean_sil = {k: np.mean(v) for k, v in _sil_matrix.items()}
_mean_ari = {k: np.mean(v) for k, v in _ari_scores.items()}

_sil_vals = np.array([_mean_sil[k] for k in _K_VALUES])
_ari_vals = np.array([_mean_ari[k] for k in _K_VALUES])

_sil_norm = (_sil_vals - _sil_vals.min()) / (_sil_vals.ptp() + 1e-9)
_ari_norm = (_ari_vals - _ari_vals.min()) / (_ari_vals.ptp() + 1e-9)
_composite = 0.6 * _sil_norm + 0.4 * _ari_norm

_best_idx = int(np.argmax(_composite))
cluster_best_k = _K_VALUES[_best_idx]

print("\n" + "="*65)
print("🏆 MODEL SELECTION SUMMARY")
print("="*65)
print(f"   {'k':>4s}  {'Silhouette':>12s}  {'ARI':>8s}  {'Composite':>10s}")
print(f"   {'-'*40}")
for _i, _k in enumerate(_K_VALUES):
    _marker = " ← BEST" if _k == cluster_best_k else ""
    print(f"   {_k:>4d}  {_mean_sil[_k]:>12.4f}  {_mean_ari[_k]:>8.4f}  {_composite[_i]:>10.4f}{_marker}")

# ═════════════════════════════════════════════════════════════════
# 4. FIT FINAL KMEANS WITH BEST k
# ═════════════════════════════════════════════════════════════════
_km_final = KMeans(n_clusters=cluster_best_k, random_state=42, n_init=30, max_iter=500)
_labels_final = _km_final.fit_predict(_X_scaled)

# Add to working dataframe
_df = _df.copy()
_df["cluster_id"] = _labels_final

# ═════════════════════════════════════════════════════════════════
# 5. CENTROID PROFILES — in original feature space
# ═════════════════════════════════════════════════════════════════
_centroids_orig = _scaler.inverse_transform(_km_final.cluster_centers_)
_centroid_df = pd.DataFrame(_centroids_orig, columns=_CLUSTER_COLS)
_centroid_df["cluster_id"] = range(cluster_best_k)
_centroid_df["n_users"]    = [(_labels_final == _c).sum() for _c in range(cluster_best_k)]

print("\n" + "="*65)
print("📊 CENTROID PROFILES (original scale)")
print("="*65)
_feat_labels = ["Sessions", "Exec Ratio", "Agent Ratio", "Onboard Ratio", "Active Days"]
print(f"   {'Cluster':>8s}  {'N':>6s}  " + "  ".join(f"{l:>12s}" for l in _feat_labels))
print(f"   {'-'*80}")
for _c in range(cluster_best_k):
    _row = _centroid_df[_centroid_df["cluster_id"] == _c].iloc[0]
    _vals = [_row[col] for col in _CLUSTER_COLS]
    _n    = int(_row["n_users"])
    print(f"   {'C' + str(_c):>8s}  {_n:>6d}  " + "  ".join(f"{v:>12.3f}" for v in _vals))

# ═════════════════════════════════════════════════════════════════
# 6. ASSIGN ARCHETYPE LABELS — data-driven from centroid profiles
# ═════════════════════════════════════════════════════════════════
# Sort clusters by activity composite: sessions * active_days (engagement signal)
_engagement = _centroid_df["feat_n_sessions"] * _centroid_df["feat_active_days"]
_sorted_ids = _engagement.argsort().values  # ascending: least → most active

cluster_archetype_names = {}
for _rank, _c in enumerate(_sorted_ids):
    _row       = _centroid_df[_centroid_df["cluster_id"] == _c].iloc[0]
    _sessions  = _row["feat_n_sessions"]
    _exec_r    = _row["feat_ratio_block_ops"]
    _agent_r   = _row["feat_ratio_agent"]
    _onboard_r = _row["feat_ratio_onboarding"]
    _days      = _row["feat_active_days"]
    _n         = int(_row["n_users"])

    # Primary signal hierarchy
    if _sessions <= 1.5 and _days <= 1.1 and _agent_r > 0.5:
        _name = "AI Sampler"          # single session, mostly agent, low engagement
    elif _sessions <= 2.5 and _onboard_r > 0.25:
        _name = "Onboarding Visitor"  # mostly onboarding events, low engagement
    elif _sessions <= 2.5 and _days <= 1.3:
        _name = "Casual Browser"      # low sessions + days, no dominant behavior
    elif _agent_r > 0.45 and _sessions > 2:
        _name = "AI-First Explorer"   # agent-driven, returning user
    elif _exec_r > 0.12 and _days >= 2:
        _name = "Hands-On Builder"    # block-execution heavy, multi-day
    elif _days >= 3:
        _name = "Power User"          # high engagement, multi-day habit
    else:
        _name = f"Mid-Tier Engager"   # moderate engagement tier

    cluster_archetype_names[_c] = _name
    _pct = _n / len(_df) * 100
    print(f"\n🏷️  Cluster {_c} → '{_name}'  (n={_n}, {_pct:.1f}%)")
    print(f"     sessions={_sessions:.1f}, exec%={_exec_r*100:.0f}%, agent%={_agent_r*100:.0f}%, "
          f"onboard%={_onboard_r*100:.0f}%, days={_days:.1f}")

_df["archetype"] = _df["cluster_id"].map(cluster_archetype_names)
print(f"\n✅ Archetype labels assigned to all {len(_df):,} users")

# ═════════════════════════════════════════════════════════════════
# 7. OUTCOME RATES BY ARCHETYPE
# ═════════════════════════════════════════════════════════════════
cluster_outcome_rates = _df.groupby("archetype").agg(
    n_users   = ("user_id_canon", "count"),
    ret30d    = ("y_ret_30d",     "mean"),
    ret90d    = ("y_ret_90d",     "mean"),
    upg60d    = ("y_upgrade_60d", "mean"),
).reset_index()

# Add cluster_id for reference
_archetype_to_cluster = {v: k for k, v in cluster_archetype_names.items()}
cluster_outcome_rates["cluster_id"] = cluster_outcome_rates["archetype"].map(_archetype_to_cluster)
cluster_outcome_rates["silhouette_k_best"] = round(_mean_sil[cluster_best_k], 4)
cluster_outcome_rates = cluster_outcome_rates.sort_values("ret90d", ascending=False).reset_index(drop=True)

print("\n" + "="*75)
print("📊 PER-CLUSTER OUTCOME RATES")
print("="*75)
print(f"{'Archetype':30s} {'N':>6s} {'ret30d':>8s} {'ret90d':>8s} {'upg60d':>8s}")
print("-"*75)
for _, _r in cluster_outcome_rates.iterrows():
    print(f"{_r['archetype']:30s} {int(_r['n_users']):>6d} "
          f"{_r['ret30d']*100:>7.1f}% {_r['ret90d']*100:>7.1f}% {_r['upg60d']*100:>7.1f}%")

_ov30 = _df["y_ret_30d"].mean()
_ov90 = _df["y_ret_90d"].mean()
_ovup = _df["y_upgrade_60d"].mean()
print("-"*75)
print(f"{'OVERALL':30s} {len(_df):>6d} {_ov30*100:>7.1f}% {_ov90*100:>7.1f}% {_ovup*100:>7.1f}%")

# ═════════════════════════════════════════════════════════════════
# 8. VISUALIZATION — Silhouette + ARI stability + Outcomes
# ═════════════════════════════════════════════════════════════════

# --- Chart 1: Silhouette & ARI across k ---
fig_silhouette_stability, (_ax_sil, _ax_ari) = plt.subplots(1, 2, figsize=(14, 6), facecolor=_BG)

for _ax in (_ax_sil, _ax_ari):
    _ax.set_facecolor(_BG)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["bottom"].set_color(_TXT2)
    _ax.spines["left"].set_color(_TXT2)
    _ax.tick_params(colors=_TXT2, labelsize=11)

# Silhouette boxplot per k
_sil_data = [_sil_matrix[k] for k in _K_VALUES]
_bp = _ax_sil.boxplot(_sil_data, labels=[f"k={k}" for k in _K_VALUES],
                       patch_artist=True, medianprops=dict(color=_HL, linewidth=2.5))
for _patch, _col in zip(_bp["boxes"], [_COLS[0], _COLS[1], _COLS[2]]):
    _patch.set_facecolor(_col)
    _patch.set_alpha(0.75)
for _wh in _bp["whiskers"] + _bp["caps"] + _bp["fliers"]:
    _wh.set_color(_TXT2)

# Highlight best k
_best_x = _K_VALUES.index(cluster_best_k) + 1
_ax_sil.axvline(_best_x, color=_HL, linewidth=1.5, linestyle="--", alpha=0.7)
_ax_sil.set_title("Silhouette Score by k (5 seeds)", color=_TXT, fontsize=13, fontweight="bold", pad=12)
_ax_sil.set_ylabel("Silhouette Score", color=_TXT, fontsize=12)
_ax_sil.set_xlabel("Number of Clusters", color=_TXT, fontsize=12)

# ARI consistency
_ari_data = [_ari_scores[k] for k in _K_VALUES]
_bp2 = _ax_ari.boxplot(_ari_data, labels=[f"k={k}" for k in _K_VALUES],
                        patch_artist=True, medianprops=dict(color=_HL, linewidth=2.5))
for _patch, _col in zip(_bp2["boxes"], [_COLS[0], _COLS[1], _COLS[2]]):
    _patch.set_facecolor(_col)
    _patch.set_alpha(0.75)
for _wh in _bp2["whiskers"] + _bp2["caps"] + _bp2["fliers"]:
    _wh.set_color(_TXT2)

_ax_ari.axvline(_best_x, color=_HL, linewidth=1.5, linestyle="--", alpha=0.7)
_ax_ari.set_title("Assignment Consistency (ARI)", color=_TXT, fontsize=13, fontweight="bold", pad=12)
_ax_ari.set_ylabel("Adjusted Rand Index", color=_TXT, fontsize=12)
_ax_ari.set_xlabel("Number of Clusters", color=_TXT, fontsize=12)

plt.suptitle(f"KMeans Stability Evaluation — Best k={cluster_best_k}  (sil={_mean_sil[cluster_best_k]:.4f})",
             color=_TXT, fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# --- Chart 2: Outcome rates by archetype ---
fig_archetype_outcomes, _axo = plt.subplots(figsize=(14, 7), facecolor=_BG)
_axo.set_facecolor(_BG)

_archetypes = cluster_outcome_rates["archetype"].values
_x = np.arange(len(_archetypes))
_width = 0.26

_metrics_map = {"ret30d": "30-Day Retention", "ret90d": "90-Day Retention", "upg60d": "60-Day Upgrade"}
for _mi, (_col_name, _label) in enumerate(_metrics_map.items()):
    _vals = cluster_outcome_rates[_col_name].values * 100
    _offset = (_mi - 1) * _width
    _bars = _axo.bar(_x + _offset, _vals, _width, label=_label,
                     color=_COLS[_mi], edgecolor="none", alpha=0.9)
    for _bar, _v in zip(_bars, _vals):
        if _v > 0.5:
            _axo.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.4,
                      f"{_v:.1f}%", ha="center", va="bottom", color=_TXT, fontsize=8, fontweight="bold")

_axo.set_xticks(_x)
_axo.set_xticklabels(_archetypes, rotation=20, ha="right", fontsize=10, color=_TXT)
_axo.set_ylabel("Rate (%)", color=_TXT, fontsize=12)
_axo.set_title(f"Retention & Upgrade Rates by User Archetype  (k={cluster_best_k})",
               color=_TXT, fontsize=14, fontweight="bold", pad=15)
_axo.tick_params(colors=_TXT2, labelsize=10)
_axo.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=10)
_axo.spines["top"].set_visible(False)
_axo.spines["right"].set_visible(False)
_axo.spines["bottom"].set_color(_TXT2)
_axo.spines["left"].set_color(_TXT2)

# Overall reference lines
_axo.axhline(_ov30 * 100, color=_COLS[0], linestyle="--", alpha=0.35, linewidth=1)
_axo.axhline(_ov90 * 100, color=_COLS[1], linestyle="--", alpha=0.35, linewidth=1)
_axo.axhline(_ovup * 100, color=_COLS[2], linestyle="--", alpha=0.35, linewidth=1)

plt.tight_layout()
plt.show()

# --- Chart 3: Feature profile radar (spider) per archetype ---
# Use bar chart per feature for clarity
_n_features = len(_CLUSTER_COLS)
_feat_display = ["Sessions", "Exec Ratio", "Agent Ratio", "Onboard Ratio", "Active Days"]

# Normalize centroids for display (0-1 scale per feature)
_centroids_norm = _centroid_df[_CLUSTER_COLS].copy()
for _col in _CLUSTER_COLS:
    _min, _max = _centroids_norm[_col].min(), _centroids_norm[_col].max()
    _centroids_norm[_col] = (_centroids_norm[_col] - _min) / (_max - _min + 1e-9)

fig_archetype_profiles, _axp = plt.subplots(figsize=(14, 6), facecolor=_BG)
_axp.set_facecolor(_BG)

_n_archetypes = cluster_best_k
_bar_width = 0.8 / _n_archetypes
_x_feat = np.arange(_n_features)

for _ci, _c in enumerate(range(cluster_best_k)):
    _arch_name = cluster_archetype_names[_c]
    _norm_vals = _centroids_norm[_centroids_norm.index == _c][_CLUSTER_COLS].values.flatten()
    _offset = (_ci - _n_archetypes / 2 + 0.5) * _bar_width
    _axp.bar(_x_feat + _offset, _norm_vals, _bar_width,
             label=_arch_name, color=_COLS[_ci % len(_COLS)], edgecolor="none", alpha=0.88)

_axp.set_xticks(_x_feat)
_axp.set_xticklabels(_feat_display, fontsize=11, color=_TXT)
_axp.set_ylabel("Normalized Centroid Value (0-1)", color=_TXT, fontsize=11)
_axp.set_title("Behavioral Feature Profiles by Archetype", color=_TXT, fontsize=14, fontweight="bold", pad=15)
_axp.tick_params(colors=_TXT2, labelsize=10)
_axp.legend(facecolor=_BG, edgecolor=_TXT2, labelcolor=_TXT, fontsize=9, ncol=2)
_axp.spines["top"].set_visible(False)
_axp.spines["right"].set_visible(False)
_axp.spines["bottom"].set_color(_TXT2)
_axp.spines["left"].set_color(_TXT2)

plt.tight_layout()
plt.show()

print(f"\n✅ KMeans archetype clustering complete.")
print(f"   • Best k = {cluster_best_k}  (silhouette = {_mean_sil[cluster_best_k]:.4f})")
print(f"   • Archetype labels assigned to all {len(_df):,} users")
print(f"   • Outcome table: {len(cluster_outcome_rates)} rows × {len(cluster_outcome_rates.columns)} cols")
