
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

_BG = "#1D1D20"
_TXT = "#fbfbff"
_TXT2 = "#909094"
_COLS = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
         "#1F77B4", "#9467BD", "#8C564B", "#C49C94", "#E377C2"]
_HL = "#ffd400"
_SUCCESS = "#17b26a"
_WARN = "#f04438"

# ═════════════════════════════════════════════════════
# 0. SETUP
# ═════════════════════════════════════════════════════
_TARGETS = ["y_ret_30d", "y_ret_90d", "y_upgrade_60d"]
_META_COLS = ["user_id_canon", "split"] + _TARGETS
_FEATURE_COLS = [c for c in modeling_df.columns if c not in _META_COLS]

_df = modeling_df.copy()

# ═════════════════════════════════════════════════════
# 1. DEFINE KEY BEHAVIORS FOR IMPACT ANALYSIS
# ═════════════════════════════════════════════════════
# Behavior definitions (binary treatment indicators from early-window features)
_BEHAVIORS = {
    "Early Deploy": {
        "treatment_col": "feat_early_deploy_count",
        "threshold": 0,  # >0 means they deployed
        "description": "Deployed (scheduled job/app publish) in first 7 days",
    },
    "Early Collaboration": {
        "treatment_col": "feat_collab_actions",
        "threshold": 0,  # >0 means they collaborated
        "description": "Shared canvas or referred in first 7 days",
    },
    "High Execution": {
        "treatment_col": "feat_n_sessions",
        "threshold": 3,  # >3 sessions = high execution (top ~30%)
        "description": "Had >3 sessions in first 7 days",
    },
}

# Confounders: everything except the treatment column and targets
_CONFOUNDER_BASE = [c for c in _FEATURE_COLS 
                    if not c.startswith("feat_primary_country")  # avoid high-dim dummies
                    and c not in ["feat_signup_dow", "feat_signup_hour"]]

print("=" * 80)
print("🔬 OBSERVATIONAL IMPACT ANALYSIS (Propensity Score Stratification)")
print("=" * 80)
print(f"\nUsing {len(_df)} users for analysis")

# ═════════════════════════════════════════════════════
# 2. PROPENSITY SCORE MATCHING/STRATIFICATION
# ═════════════════════════════════════════════════════
impact_results = []

for _beh_name, _beh_config in _BEHAVIORS.items():
    _treat_col = _beh_config["treatment_col"]
    _thresh = _beh_config["threshold"]
    _desc = _beh_config["description"]
    
    # Create binary treatment
    _treatment = (_df[_treat_col] > _thresh).astype(int).values
    _n_treated = _treatment.sum()
    _n_control = len(_treatment) - _n_treated
    
    print(f"\n{'─'*70}")
    print(f"🎯 BEHAVIOR: {_beh_name}")
    print(f"   {_desc}")
    print(f"   Treated: {_n_treated} ({_n_treated/len(_treatment)*100:.1f}%)")
    print(f"   Control: {_n_control} ({_n_control/len(_treatment)*100:.1f}%)")
    
    if _n_treated < 10 or _n_control < 10:
        print(f"   ⚠️  SKIPPED: insufficient sample size for reliable estimates")
        impact_results.append({
            "Behavior": _beh_name,
            "N_Treated": _n_treated,
            "N_Control": _n_control,
            "Target": "N/A",
            "Raw_Diff": np.nan,
            "Matched_Diff": np.nan,
            "Signal_Strength": "Insufficient Data",
        })
        continue
    
    # Build confounders (exclude treatment col itself)
    _confounders = [c for c in _CONFOUNDER_BASE if c != _treat_col]
    _X_conf = _df[_confounders].values
    
    # Scale for logistic regression
    _scaler = StandardScaler()
    _X_conf_scaled = _scaler.fit_transform(_X_conf)
    
    # Fit propensity score model
    _ps_model = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    _ps_model.fit(_X_conf_scaled, _treatment)
    _propensity = _ps_model.predict_proba(_X_conf_scaled)[:, 1]
    
    # Trim propensity scores to avoid extreme weights
    _ps_trimmed = np.clip(_propensity, 0.05, 0.95)
    
    # Propensity score stratification (5 strata)
    _N_STRATA = 5
    _ps_quantiles = np.percentile(_ps_trimmed, np.linspace(0, 100, _N_STRATA + 1))
    _ps_quantiles[0] = _ps_trimmed.min() - 0.001
    _ps_quantiles[-1] = _ps_trimmed.max() + 0.001
    _strata = np.digitize(_ps_trimmed, _ps_quantiles) - 1
    _strata = np.clip(_strata, 0, _N_STRATA - 1)
    
    for _target in _TARGETS:
        _y = _df[_target].values
        
        # Raw (unadjusted) difference
        _raw_treated_rate = _y[_treatment == 1].mean()
        _raw_control_rate = _y[_treatment == 0].mean()
        _raw_diff = _raw_treated_rate - _raw_control_rate
        
        # Stratified estimate (weighted average of within-stratum differences)
        _strat_diffs = []
        _strat_weights = []
        _valid_strata = 0
        
        for _s in range(_N_STRATA):
            _mask_s = _strata == _s
            _treat_s = _treatment[_mask_s]
            _y_s = _y[_mask_s]
            
            _n_t_s = (_treat_s == 1).sum()
            _n_c_s = (_treat_s == 0).sum()
            
            if _n_t_s >= 3 and _n_c_s >= 3:
                _diff_s = _y_s[_treat_s == 1].mean() - _y_s[_treat_s == 0].mean()
                _strat_diffs.append(_diff_s)
                _strat_weights.append(_mask_s.sum())
                _valid_strata += 1
        
        if _valid_strata >= 2:
            _strat_weights = np.array(_strat_weights, dtype=float)
            _strat_weights /= _strat_weights.sum()
            _matched_diff = np.average(_strat_diffs, weights=_strat_weights)
        else:
            _matched_diff = _raw_diff  # fallback
        
        # Bootstrap CI for matched difference
        _n_boot = 500
        _boot_diffs = []
        _rng = np.random.RandomState(42)
        for _b in range(_n_boot):
            _idx = _rng.choice(len(_y), size=len(_y), replace=True)
            _b_treat = _treatment[_idx]
            _b_y = _y[_idx]
            _b_strata = _strata[_idx]
            
            _b_diffs = []
            _b_weights = []
            for _s in range(_N_STRATA):
                _mask_s = _b_strata == _s
                _b_treat_s = _b_treat[_mask_s]
                _b_y_s = _b_y[_mask_s]
                _n_t_s = (_b_treat_s == 1).sum()
                _n_c_s = (_b_treat_s == 0).sum()
                if _n_t_s >= 2 and _n_c_s >= 2:
                    _d = _b_y_s[_b_treat_s == 1].mean() - _b_y_s[_b_treat_s == 0].mean()
                    _b_diffs.append(_d)
                    _b_weights.append(_mask_s.sum())
            
            if len(_b_diffs) >= 2:
                _b_w = np.array(_b_weights, dtype=float)
                _b_w /= _b_w.sum()
                _boot_diffs.append(np.average(_b_diffs, weights=_b_w))
        
        _ci_low = np.percentile(_boot_diffs, 2.5) if _boot_diffs else np.nan
        _ci_high = np.percentile(_boot_diffs, 97.5) if _boot_diffs else np.nan
        
        # Signal classification
        if np.isnan(_ci_low) or np.isnan(_ci_high):
            _signal = "Insufficient Data"
        elif _ci_low > 0:
            _signal = "🟢 STRONG (CI excludes 0)"
        elif _ci_high < 0:
            _signal = "🟢 STRONG NEGATIVE"
        elif abs(_matched_diff) > 0.02:
            _signal = "🟡 TENTATIVE (suggestive)"
        else:
            _signal = "⚪ WEAK (near zero)"
        
        _target_lbl = {"y_ret_30d": "Ret30d", "y_ret_90d": "Ret90d", "y_upgrade_60d": "Upg60d"}[_target]
        
        print(f"\n   📊 {_target_lbl}:")
        print(f"      Raw diff:     {_raw_diff*100:+.1f}pp  (treated={_raw_treated_rate*100:.1f}%, control={_raw_control_rate*100:.1f}%)")
        print(f"      Matched diff: {_matched_diff*100:+.1f}pp  [{_ci_low*100:+.1f}pp, {_ci_high*100:+.1f}pp]  {_signal}")
        
        impact_results.append({
            "Behavior": _beh_name,
            "N_Treated": _n_treated,
            "N_Control": _n_control,
            "Target": _target_lbl,
            "Raw_Diff_pp": _raw_diff * 100,
            "Matched_Diff_pp": _matched_diff * 100,
            "CI_Low_pp": _ci_low * 100,
            "CI_High_pp": _ci_high * 100,
            "Signal_Strength": _signal,
        })

impact_df = pd.DataFrame(impact_results)

# ═════════════════════════════════════════════════════
# 3. IMPACT VISUALIZATION — forest plot
# ═════════════════════════════════════════════════════
_valid_impact = impact_df.dropna(subset=["Matched_Diff_pp"])
_valid_impact = _valid_impact[_valid_impact["Target"] != "N/A"].copy()

fig_impact_forest, _ax = plt.subplots(figsize=(14, 8), facecolor=_BG)
_ax.set_facecolor(_BG)

_y_positions = np.arange(len(_valid_impact))[::-1]
_labels = [f"{r['Behavior']} → {r['Target']}" for _, r in _valid_impact.iterrows()]

for _i, (_, _row) in enumerate(_valid_impact.iterrows()):
    _y = _y_positions[_i]
    _diff = _row["Matched_Diff_pp"]
    _ci_l = _row["CI_Low_pp"]
    _ci_h = _row["CI_High_pp"]
    
    # Color by signal
    if "STRONG" in str(_row["Signal_Strength"]) and "NEGATIVE" not in str(_row["Signal_Strength"]):
        _color = _SUCCESS
    elif "TENTATIVE" in str(_row["Signal_Strength"]):
        _color = _HL
    else:
        _color = _TXT2
    
    _ax.plot([_ci_l, _ci_h], [_y, _y], color=_color, linewidth=2.5, alpha=0.7)
    _ax.scatter([_diff], [_y], color=_color, s=80, zorder=5, edgecolors="white", linewidth=0.5)

_ax.axvline(0, color=_TXT2, linestyle="--", linewidth=1, alpha=0.5)
_ax.set_yticks(_y_positions)
_ax.set_yticklabels(_labels, fontsize=10, color=_TXT)
_ax.set_xlabel("Matched Difference (percentage points)", color=_TXT, fontsize=12)
_ax.set_title("Observational Impact Estimates (Propensity Score Stratification)\n95% Bootstrap CI",
              color=_TXT, fontsize=14, fontweight="bold", pad=15)
_ax.tick_params(colors=_TXT2, labelsize=10)
_ax.spines["top"].set_visible(False)
_ax.spines["right"].set_visible(False)
_ax.spines["bottom"].set_color(_TXT2)
_ax.spines["left"].set_color(_TXT2)

plt.tight_layout()
plt.show()

# ═════════════════════════════════════════════════════
# 4. UNDERPERFORMING SEGMENTS
# ═════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("📉 UNDERPERFORMING SEGMENTS")
print("=" * 80)

_overall_90 = _df["y_ret_90d"].mean()
_overall_up = _df["y_upgrade_60d"].mean()

# Segment 1: Casual visitors with AI intent (agent-only, no block execution)
_seg1 = _df[(_df["feat_ratio_agent"] > 0.5) & (_df["feat_ratio_block_ops"] == 0) & (_df["feat_event_count"] > 3)]
_seg1_ret90 = _seg1["y_ret_90d"].mean()
_seg1_up = _seg1["y_upgrade_60d"].mean()
print(f"\n1️⃣  AI-Only Browsers (agent>50%, zero block ops, >3 events)")
print(f"   Size: {len(_seg1)} users ({len(_seg1)/len(_df)*100:.1f}%)")
print(f"   Ret90d: {_seg1_ret90*100:.1f}% vs {_overall_90*100:.1f}% overall (Δ={(_seg1_ret90-_overall_90)*100:+.1f}pp)")
print(f"   Upg60d: {_seg1_up*100:.1f}% vs {_overall_up*100:.1f}% overall (Δ={(_seg1_up-_overall_up)*100:+.1f}pp)")
print(f"   🔍 Insight: Users engaging only with AI assistant but not building anything")

# Segment 2: Onboarding completers who never return  
_seg2 = _df[(_df["feat_onboarding_completed"] == 1) & (_df["feat_active_days"] <= 1) & (_df["feat_n_sessions"] <= 2)]
_seg2_ret90 = _seg2["y_ret_90d"].mean()
_seg2_up = _seg2["y_upgrade_60d"].mean()
print(f"\n2️⃣  Onboarding Drop-Offs (completed onboarding, 1 day, ≤2 sessions)")
print(f"   Size: {len(_seg2)} users ({len(_seg2)/len(_df)*100:.1f}%)")
print(f"   Ret90d: {_seg2_ret90*100:.1f}% vs {_overall_90*100:.1f}% overall (Δ={(_seg2_ret90-_overall_90)*100:+.1f}pp)")
print(f"   Upg60d: {_seg2_up*100:.1f}% vs {_overall_up*100:.1f}% overall (Δ={(_seg2_up-_overall_up)*100:+.1f}pp)")
print(f"   🔍 Insight: Completed the tour but didn't find compelling use-case")

# Segment 3: Multi-session users who never run blocks
_seg3 = _df[(_df["feat_n_sessions"] >= 2) & (_df["feat_ttf_run_block"] >= 999)]
_seg3_ret90 = _seg3["y_ret_90d"].mean()
_seg3_up = _seg3["y_upgrade_60d"].mean()
print(f"\n3️⃣  Browsers Without Execution (≥2 sessions, never ran a block)")
print(f"   Size: {len(_seg3)} users ({len(_seg3)/len(_df)*100:.1f}%)")
print(f"   Ret90d: {_seg3_ret90*100:.1f}% vs {_overall_90*100:.1f}% overall (Δ={(_seg3_ret90-_overall_90)*100:+.1f}pp)")
print(f"   Upg60d: {_seg3_up*100:.1f}% vs {_overall_up*100:.1f}% overall (Δ={(_seg3_up-_overall_up)*100:+.1f}pp)")
print(f"   🔍 Insight: Interested enough to return, but never executed code — friction point")

# ═════════════════════════════════════════════════════
# 5. SUMMARY TABLE
# ═════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("📊 COMPLETE IMPACT SUMMARY TABLE")
print("=" * 80)
print(f"\n{'Behavior':22s} {'Target':8s} {'N_T':>6s} {'N_C':>6s} {'Raw Δ':>8s} {'Match Δ':>8s} {'95% CI':>16s} {'Signal':>30s}")
print("-" * 110)
for _, _r in impact_df.iterrows():
    if _r["Target"] == "N/A":
        continue
    _ci = f"[{_r['CI_Low_pp']:+.1f}, {_r['CI_High_pp']:+.1f}]" if not np.isnan(_r.get("CI_Low_pp", np.nan)) else "N/A"
    print(f"{_r['Behavior']:22s} {_r['Target']:8s} {_r['N_Treated']:6.0f} {_r['N_Control']:6.0f} "
          f"{_r['Raw_Diff_pp']:+7.1f}pp {_r['Matched_Diff_pp']:+7.1f}pp {_ci:>16s} {_r['Signal_Strength']:>30s}")

print("\n✅ Propensity score impact analysis complete.")
