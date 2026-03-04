
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════
BG   = "#1D1D20"
TXT  = "#fbfbff"
TXT2 = "#909094"
COLS = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
        "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
HL   = "#ffd400"
GREEN = "#17b26a"
WARN  = "#f04438"

# ═══════════════════════════════════════════════════════════════
# 0. BUILD USER-LEVEL DATASET
# ═══════════════════════════════════════════════════════════════
_CLUSTER_COLS = [
    "feat_n_sessions", "feat_ratio_block_ops",
    "feat_ratio_agent", "feat_ratio_onboarding", "feat_active_days",
]

# Refit KMeans with same 5 features to get per-user assignments
_UPLIFT_K = len(cluster_archetype_names)  # 5 — avoids cluster_best_k conflict
_scaler_up = StandardScaler()
_X_up_scaled = _scaler_up.fit_transform(modeling_df[_CLUSTER_COLS].values)
_km_up = KMeans(n_clusters=_UPLIFT_K, random_state=42, n_init=30, max_iter=500)
_km_up.fit(_X_up_scaled)
_user_cluster_ids = _km_up.predict(_X_up_scaled)

_user_df = modeling_df[[
    "user_id_canon",
    "feat_event_count", "feat_n_sessions", "feat_active_days",
    "feat_ttf_run_block", "feat_ttf_agent_use",
    "feat_ratio_agent", "feat_ratio_block_ops", "feat_ratio_onboarding",
    "feat_onboarding_completed", "feat_onboarding_skipped",
    "feat_collab_actions", "feat_early_deploy_count",
    "y_ret_30d", "y_ret_90d", "y_upgrade_60d"
]].copy()
_user_df["cluster_id"] = _user_cluster_ids
_user_df["archetype"]  = _user_df["cluster_id"].map(cluster_archetype_names)
_user_df["agent_usage_ratio"] = _user_df["feat_ratio_agent"]

# Predicted Risk (logistic regression on ret_30d as churn-risk proxy)
_FEAT_RISK = [c for c in modeling_df.columns if c.startswith("feat_") and c in _user_df.columns]
_scaler_risk = StandardScaler()
_X_risk_s = _scaler_risk.fit_transform(_user_df[_FEAT_RISK].fillna(0).values)
_lr_risk = LogisticRegression(C=0.5, max_iter=500, random_state=42, class_weight="balanced")
_lr_risk.fit(_X_risk_s, _user_df["y_ret_30d"].values)
_user_df["predicted_risk"] = _lr_risk.predict_proba(_X_risk_s)[:, 1]

print("═"*70)
print("📊 USER-LEVEL DATASET  (archetype + predicted_risk + agent_usage_ratio)")
print("═"*70)
print(f"   Total users: {len(_user_df):,}")
print(f"\n   {'Archetype':25s} {'N':>6s} {'%':>5s}  {'avg_risk':>9s}  {'agent%':>7s}")
print("-"*60)
for _arch, _cnt in _user_df["archetype"].value_counts().items():
    _sub = _user_df[_user_df["archetype"] == _arch]
    print(f"   {_arch:25s} {_cnt:6d} {_cnt/len(_user_df)*100:5.1f}%  "
          f"{_sub['predicted_risk'].mean():9.3f}  {_sub['agent_usage_ratio'].mean()*100:6.0f}%")

# ═══════════════════════════════════════════════════════════════
# 1. ARCHETYPE BASELINES + IMPACT SIGNALS
# ═══════════════════════════════════════════════════════════════
_ARCHETYPE_BASELINES = {
    row["archetype"]: {
        "n_users": row["n_users"], "ret30d": row["ret30d"],
        "ret90d": row["ret90d"],   "upg60d": row["upg60d"],
    }
    for _, row in cluster_outcome_rates.iterrows()
}

print("\n" + "═"*70)
print("📊 ARCHETYPE BASELINES  (source: kmeans_archetype_clustering)")
print("═"*70)
print(f"{'Archetype':30s} {'N':>6s} {'ret30d':>8s} {'ret90d':>8s} {'upg60d':>8s}")
print("-"*60)
for _arch, _b in sorted(_ARCHETYPE_BASELINES.items(), key=lambda x: -x[1]["ret90d"]):
    print(f"{_arch:30s} {_b['n_users']:6d} {_b['ret30d']*100:7.1f}% "
          f"{_b['ret90d']*100:7.1f}% {_b['upg60d']*100:7.1f}%")

# Extract propensity uplift signals
_impact_map = {}
for _, _row in impact_df.iterrows():
    if _row["Target"] != "N/A":
        _key = (_row["Behavior"], _row["Target"])
        _val = _row["Matched_Diff_pp"]
        if np.isnan(_val):
            _val = _row["Raw_Diff_pp"]
        _impact_map[_key] = float(_val) if not np.isnan(_val) else 0.0

_exec_uplift_ret30   = _impact_map.get(("High Execution", "Ret30d"), 18.0)
_exec_uplift_ret90   = _impact_map.get(("High Execution", "Ret90d"), 25.0)
_collab_uplift_ret90 = _impact_map.get(("Early Collaboration", "Ret90d"), 25.2)
_DISCOUNT = 0.45  # observational → causal discount

print("\n" + "═"*70)
print("📊 PROPENSITY UPLIFT SIGNALS  (from propensity_impact_analysis)")
print("═"*70)
print(f"   High Execution → Ret30d : +{_exec_uplift_ret30:.1f}pp (×{_DISCOUNT} causal discount)")
print(f"   High Execution → Ret90d : +{_exec_uplift_ret90:.1f}pp")
print(f"   Early Collab   → Ret90d : +{_collab_uplift_ret90:.1f}pp")

# ═══════════════════════════════════════════════════════════════
# 2. INTERVENTION DEFINITIONS
# ═══════════════════════════════════════════════════════════════
INTERVENTIONS = [
    {
        "name": "onboarding_to_build_nudge",
        "label": "Onboarding→Build Nudge",
        "description": "In-product prompt at onboarding completion guiding users to run their first block",
        "mechanism": "Bridges onboarding→execution gap (feat_ttf_run_block acceleration)",
        "primary_target": "Onboarding Visitor",
        "secondary_target": "Casual Browser",
        "target_metric": "ret30d",
        "uplift_by_archetype": {
            "Onboarding Visitor": _exec_uplift_ret30 * _DISCOUNT * 0.80,
            "Casual Browser":     _exec_uplift_ret30 * _DISCOUNT * 0.50,
            "Mid-Tier Engager":   _exec_uplift_ret30 * _DISCOUNT * 0.20,
            "Hands-On Builder":   0.5,
        },
        "engineering_cost": 1, "time_to_impact": 1, "confidence": 0.72,
    },
    {
        "name": "agent_to_block_conversion_UI_flow",
        "label": "Agent→Block Conversion UI",
        "description": "UI flow surfacing block creation from agent output (85%+ agent-usage users)",
        "mechanism": "Converts AI-only sessions into block execution; reduces feat_ratio_agent dominance",
        "primary_target": "Casual Browser",
        "secondary_target": "Mid-Tier Engager",
        "target_metric": "ret90d",
        "uplift_by_archetype": {
            "Casual Browser":     _exec_uplift_ret30 * _DISCOUNT * 1.00,
            "Mid-Tier Engager":   _exec_uplift_ret30 * _DISCOUNT * 0.40,
            "Onboarding Visitor": _exec_uplift_ret30 * _DISCOUNT * 0.20,
            "Hands-On Builder":   0.5,
        },
        "engineering_cost": 2, "time_to_impact": 2, "confidence": 0.68,
    },
    {
        "name": "session_milestone_checklist",
        "label": "Session Milestone Checklist",
        "description": "Per-session progress checklist: run block, connect data, deploy, collaborate",
        "mechanism": "Drives multi-session habit (feat_n_sessions ↑, feat_active_days ↑)",
        "primary_target": "Mid-Tier Engager",
        "secondary_target": "Onboarding Visitor",
        "target_metric": "ret90d",
        "uplift_by_archetype": {
            "Mid-Tier Engager":   _exec_uplift_ret90 * _DISCOUNT * 0.70,
            "Onboarding Visitor": _exec_uplift_ret90 * _DISCOUNT * 0.35,
            "Casual Browser":     _exec_uplift_ret90 * _DISCOUNT * 0.25,
            "Hands-On Builder":   0.5,
        },
        "engineering_cost": 1, "time_to_impact": 2, "confidence": 0.65,
    },
    {
        "name": "day1_day3_day7_email_drip",
        "label": "Day 1/3/7 Email Drip",
        "description": "Timed email sequence: Day 1 (build nudge), Day 3 (deploy), Day 7 (collaborate)",
        "mechanism": "Re-engagement across at-risk windows; surfaces collaboration + deployment",
        "primary_target": "Onboarding Visitor",
        "secondary_target": "Casual Browser",
        "target_metric": "ret90d",
        "uplift_by_archetype": {
            "Onboarding Visitor": _collab_uplift_ret90 * _DISCOUNT * 0.65,
            "Casual Browser":     _collab_uplift_ret90 * _DISCOUNT * 0.55,
            "Mid-Tier Engager":   _collab_uplift_ret90 * _DISCOUNT * 0.35,
            "Hands-On Builder":   1.0,
        },
        "engineering_cost": 1, "time_to_impact": 1, "confidence": 0.60,
    },
]

# ═══════════════════════════════════════════════════════════════
# 3. PER-USER UPLIFT SCORING
# ═══════════════════════════════════════════════════════════════
def compute_user_uplift(user_row, intervention):
    _arch = user_row["archetype"]
    _risk = user_row["predicted_risk"]
    _arch_uplift_pp = intervention["uplift_by_archetype"].get(_arch, 0.5)
    _persuadability = 4 * _risk * (1 - _risk)  # parabola peaking at risk=0.5
    _agent_bonus = 1.0 + user_row["agent_usage_ratio"] * 0.5 if "agent" in intervention["name"] else 1.0
    return max(0.0, _arch_uplift_pp * intervention["confidence"] * _persuadability * _agent_bonus)

for _iv in INTERVENTIONS:
    _colname = "uplift_" + _iv["name"]
    _user_df[_colname] = _user_df.apply(lambda r: compute_user_uplift(r, _iv), axis=1)

# Print uplift matrix — avoid f-string backslash issue
print("\n" + "═"*80)
print("📊 MEAN UPLIFT SCORES BY ARCHETYPE × INTERVENTION (pp)")
print("═"*80)
_header = f"{'Archetype':25s}" + "".join(f"{iv['label'][:16]:>18s}" for iv in INTERVENTIONS)
print(_header)
print("-"*100)
for _arch in sorted(_user_df["archetype"].dropna().unique()):
    _sub = _user_df[_user_df["archetype"] == _arch]
    _row_str = f"{_arch:25s}"
    for _iv in INTERVENTIONS:
        _upcol = "uplift_" + _iv["name"]
        _row_str += f"{_sub[_upcol].mean():>18.2f}"
    print(_row_str)

# ═══════════════════════════════════════════════════════════════
# 4. PRIORITY FORMULA: 25%×segment + 45%×uplift + 20%×eng_eff + 10%×speed
# ═══════════════════════════════════════════════════════════════
WEIGHTS = {"segment_size": 0.25, "total_uplift": 0.45, "eng_efficiency": 0.20, "speed": 0.10}

_intervention_scores = []
for _iv in INTERVENTIONS:
    _upcol = "uplift_" + _iv["name"]
    _prim_mask = _user_df["archetype"] == _iv["primary_target"]
    _sec_mask  = _user_df["archetype"] == _iv["secondary_target"]
    _n_addressable = (_prim_mask | _sec_mask).sum()
    _n_total = len(_user_df)

    _prim_n   = _ARCHETYPE_BASELINES.get(_iv["primary_target"],   {}).get("n_users", 0)
    _sec_n    = _ARCHETYPE_BASELINES.get(_iv["secondary_target"],  {}).get("n_users", 0)
    _prim_up  = _iv["uplift_by_archetype"].get(_iv["primary_target"], 0)
    _sec_up   = _iv["uplift_by_archetype"].get(_iv["secondary_target"], 0)
    _expected_add_ret = (
        _prim_n * (_prim_up / 100) * _iv["confidence"] +
        _sec_n  * (_sec_up  / 100) * _iv["confidence"]
    )

    _intervention_scores.append({
        "Intervention":          _iv["label"],
        "name":                  _iv["name"],
        "description":           _iv["description"],
        "primary_target":        _iv["primary_target"],
        "secondary_target":      _iv["secondary_target"],
        "n_addressable":         _n_addressable,
        "seg_pct":               _n_addressable / _n_total * 100,
        "mean_uplift_pp":        _user_df[_upcol].mean(),
        "median_uplift_pp":      _user_df[_upcol].median(),
        "p75_uplift_pp":         _user_df[_upcol].quantile(0.75),
        "total_uplift_sum":      _user_df[_upcol].sum(),
        "expected_add_retentions": _expected_add_ret,
        "eng_cost":              _iv["engineering_cost"],
        "time_to_impact":        _iv["time_to_impact"],
        "confidence":            _iv["confidence"],
        "segment_size_score":    _n_addressable / _n_total,
        "eng_efficiency_score":  (4 - _iv["engineering_cost"]) / 3.0,
        "speed_score":           (4 - _iv["time_to_impact"]) / 3.0,
        "mechanism":             _iv["mechanism"],
    })

_score_df = pd.DataFrame(_intervention_scores)
_max_uplift = _score_df["total_uplift_sum"].max()
_score_df["total_uplift_score_norm"] = _score_df["total_uplift_sum"] / (_max_uplift + 1e-9)

_score_df["priority_score"] = (
    WEIGHTS["segment_size"]  * _score_df["segment_size_score"] +
    WEIGHTS["total_uplift"]  * _score_df["total_uplift_score_norm"] +
    WEIGHTS["eng_efficiency"]* _score_df["eng_efficiency_score"] +
    WEIGHTS["speed"]         * _score_df["speed_score"]
)

_score_df = _score_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
_score_df["priority_rank"] = _score_df.index + 1

# Top segments per intervention
_top_segments = {}
for _iv in INTERVENTIONS:
    _upcol = "uplift_" + _iv["name"]
    _segs = (
        _user_df.groupby("archetype")[_upcol]
        .agg(n_users="count", mean_uplift="mean", total_uplift="sum")
        .reset_index().sort_values("total_uplift", ascending=False).head(3)
    )
    _top_segments[_iv["name"]] = _segs

# ═══════════════════════════════════════════════════════════════
# 5. PRINT RANKED INTERVENTION TABLE
# ═══════════════════════════════════════════════════════════════
_COST_LABEL = {1: "Low", 2: "Med", 3: "High"}
_TIME_LABEL  = {1: "Fast (<1wk)", 2: "Med (2-4wk)", 3: "Slow (>1mo)"}

print("\n" + "═"*120)
print("🏆  INTERVENTION PRIORITY RANKING")
print("     Formula: 25% × segment_size + 45% × uplift_potential + 20% × eng_efficiency + 10% × speed")
print("═"*120)
print(f"\n  {'#':>3}  {'Intervention':30s}  {'Score':>6}  {'N Users':>7}  {'Seg%':>5}  "
      f"{'AvgUp':>7}  {'Exp+Ret':>7}  {'Cost':>4}  {'Time':>12}  {'Conf':>5}")
print("─"*105)

uplift_intervention_table = _score_df.copy()

for _, _r in _score_df.iterrows():
    print(
        f"  #{int(_r['priority_rank']):2d}  "
        f"{_r['Intervention']:30s}  "
        f"{_r['priority_score']:6.4f}  "
        f"{int(_r['n_addressable']):7d}  "
        f"{_r['seg_pct']:5.1f}%  "
        f"{_r['mean_uplift_pp']:>6.2f}pp  "
        f"{_r['expected_add_retentions']:>7.1f}  "
        f"{_COST_LABEL[int(_r['eng_cost'])]:>4s}  "
        f"{_TIME_LABEL[int(_r['time_to_impact'])]:>12s}  "
        f"{_r['confidence']:>5.0%}"
    )

print("\n\n📍 TOP USER SEGMENTS PER INTERVENTION  (by total expected impact):")
print("═"*90)
for _iv in INTERVENTIONS:
    _iv_row = _score_df[_score_df["name"] == _iv["name"]].iloc[0]
    _rank = int(_iv_row["priority_rank"])
    print(f"\n  #{_rank}: {_iv['label']}  [Score: {_iv_row['priority_score']:.4f}]")
    print(f"     Desc: {_iv['description']}")
    print(f"     How:  {_iv['mechanism']}")
    _segs = _top_segments[_iv["name"]]
    for _, _s in _segs.iterrows():
        print(f"     ▸ {_s['archetype']:25s}: {int(_s['n_users']):4d} users | "
              f"mean_uplift={_s['mean_uplift']:.2f}pp | total_impact={_s['total_uplift']:.0f}pp×users")

print("\n✅ Uplift estimation complete.")
print(f"   Cohort: {len(_user_df):,} users  |  {len(INTERVENTIONS)} interventions scored")
print(f"   Weights: segment×{WEIGHTS['segment_size']} + uplift×{WEIGHTS['total_uplift']} "
      f"+ eng×{WEIGHTS['eng_efficiency']} + speed×{WEIGHTS['speed']}")
