# BuilderFlow: Written Summary

## The Question

**What early behaviors in a user's first week on Zerve predict whether they will become a long-term, paying customer?**

Every platform faces the same challenge: most signups never return. The Zerve hackathon dataset gave us 226,600 behavioral events from 1,472 users, each captured during their first days on the platform. We set out to answer a deceptively simple question: *can we tell, within 7 days, which users will retain at 30 and 90 days, and which will upgrade within 60 days?*

More importantly, we wanted to move beyond prediction into prescription: once we know *who* is at risk of churning and *why*, what specific product interventions would change their trajectory?

---

## Success Definition

We defined success across three measurable dimensions:

**1. Predictive Accuracy on Extremely Sparse Outcomes.**
The 30-day retention rate on our test set was just 1.1%. At this base rate, a naive model achieves near-perfect accuracy by predicting "no retention" for everyone, which is worthless. We therefore chose **Precision-Recall AUC (PR-AUC)** as our primary metric, because it specifically measures a model's ability to find the rare positive cases without drowning in false positives. A PR-AUC above the 0.011 baseline would indicate meaningful signal; anything above 0.10 would be operationally useful. We also tracked **Lift @ Top 10%** to answer the practical question: *if we could only intervene on 10% of users, how much better than random would our targeting be?*

**2. Interpretable, Stable Explanations.**
A black-box model that says "this user will churn" is not actionable. We required our feature importance rankings to be stable across bootstrap resamples (coefficient of variation < 0.25), so that the product team could trust the drivers and build interventions around them.

**3. Concrete, Prioritized Interventions.**
The final deliverable had to be a ranked list of product changes with estimated uplift in retained users, not just a model artifact. Success meant answering: *"Which intervention should engineering build first, and how many users will it save?"*

---

## Methodology

### Data Pipeline

We built a 21-block Zerve Canvas pipeline organized as a directed acyclic graph (DAG):

**Stage 1 — Cohort Construction.** We loaded the raw event log, parsed timestamps, created canonical user identifiers, and filtered to our analysis cohort based on a temporal cutoff. This ensured a clean, deduplicated user base.

**Stage 2 — Feature Engineering (7-Day Window).** From each user's first 7 days of activity, we engineered 51 features across five groups:
- *Intensity* — event counts, session counts, active days
- *Advanced Usage* — ratios of agent, block execution, canvas, and collaboration events
- *Time-to-First* — hours until first block run, first agent use, first canvas creation
- *Behavioral Signals* — session entropy, time-of-day entropy, day-gap variance, execution-to-agent ratio
- *Metadata* — target-encoded country, device, browser; signup hour

We applied zero-variance filtering, removed features with correlation > 0.92, and target-encoded sparse categoricals. All features are strictly from days 1–7; labels are defined from day 8 onward. No temporal leakage.

**Stage 3 — User Segmentation.** We ran KMeans clustering (k ∈ {4, 5, 6}) on five behavioral dimensions: sessions, execution ratio, agent ratio, onboarding ratio, and active days. Silhouette scoring selected k=6, and we assigned data-driven archetype names based on each cluster's behavioral profile.

**Stage 4 — Predictive Modeling.** We trained an XGBoost classifier with Bayesian hyperparameter optimization (30 trials: 15 exploration + 15 exploitation) over depth, learning rate, estimator count, subsample, and regularization. We used `scale_pos_weight` for class imbalance, temporal train/test splitting (Sep–Oct 2025 train, Nov 1–8 test), and 3-window rolling cross-validation for stability assessment. Post-training, we applied isotonic regression for probability calibration and tuned the decision threshold using F-beta (β=2) to weight recall over precision.

**Stage 5 — Interpretability.** We computed SHAP values via TreeExplainer for global feature importance, beeswarm plots, interaction analysis, and per-user force plots. We ran bootstrap variance estimation (20 resamples) to confirm SHAP stability, and performed an ablation study by dropping feature groups one at a time to measure their marginal contribution.

**Stage 6 — Uplift Estimation & Intervention Scoring.** For each of four candidate interventions, we estimated per-user uplift as a function of archetype baseline, causal discount (40%), persuadability (a parabolic function of predicted risk), and intervention-specific confidence. We then ranked interventions by a composite priority score weighting segment size (25%), total uplift potential (45%), engineering cost (20%), and speed to market (10%).

### Deployed Functionality

We added a **Scheduled Job** layer: a daily-running pipeline that re-scores all users with the calibrated model, assigns risk tiers (Low / Medium / High / Critical), and saves a timestamped CSV for the product team.

---

## Findings

### Finding 1: Active construction, not passive exploration, predicts retention.

The single strongest predictor of retention is `active_days` (mean |SHAP| = 0.229), the number of distinct days a user was active in their first week. This was followed by `max_gap_days` (0.093)—how long users go dormant between sessions—and `agent_usage_ratio` (0.085). The pattern is clear: users who *do things* (run blocks, create canvases, return on multiple days) retain. Users who only browse or only use the AI agent without executing do not.

### Finding 2: Six archetypes with dramatically different outcomes.

Our clustering revealed six behavioral archetypes. At the extremes:
- **Hands-On Builders** (1.1% of users): 81% retention at 30 and 90 days, 25% upgrade rate. These users execute blocks, create canvases, and return across multiple days.
- **Casual Browsers** (42.5% of users): 4% retention at 30 days, 1% upgrade rate. These users explore the interface but never build anything.

The gap between these groups is 19×. Every product intervention we identified aims to move users from Casual Browser behavior toward Builder behavior.

### Finding 3: The model works despite extreme sparsity.

Our calibrated XGBoost achieved PR-AUC of 0.269 against a 0.011 baseline (24× improvement), with Lift @ Top 10% of 2.56×. Rolling cross-validation confirmed stability (PR-AUC 0.249 ± 0.058). Isotonic calibration improved Brier score from 0.217 to 0.013, making the predicted probabilities trustworthy for downstream scoring.

### Finding 4: Agent usage is a double-edged sword.

`agent_usage_ratio` is the #3 SHAP driver, but its effect is *negative* at high values. Users who rely heavily on the AI agent without converting agent output into block execution show lower retention. This is the single most actionable insight: the product needs a bridge from AI-generated output to hands-on building.

### Finding 5: Four interventions, ranked by expected impact.

| Rank | Intervention | Expected New Retentions | Cost |
|------|-------------|------------------------|------|
| **#1** | **Agent→Block Conversion UI**: surface a "Run This" button on agent output | +21 users | Low |
| **#2** | **Day 1/3/7 Email Drip**: timed re-engagement sequence segmented by archetype | +20 users | Low |
| **#3** | **Session Milestone Checklist**: in-session progress tracker (run block, deploy, collaborate) | +16 users | Low |
| **#4** | **Onboarding→Build Nudge**: post-tour wizard guiding first canvas creation | +8 users | Low |

All four interventions are low-engineering-cost and fast-to-market. The top intervention alone (Agent→Block Conversion UI) addresses 73% of the user base and is projected to convert approximately 21 additional users from at-risk to retained.

---

## Closing Thesis

> **The strongest predictor of retention is not feature exposure, but the early transition from passive exploration to active construction.** Users who execute blocks, create canvases, and return across multiple days form the Hands-On Builder archetype—achieving 81% retention versus 4% for Casual Browsers. The path from churn to retention is bridging AI-generated insights into executable workflows within the first 7 days. BuilderFlow makes this bridge visible, measurable, and actionable.
