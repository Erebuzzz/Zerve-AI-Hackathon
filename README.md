# BuilderFlow: Zerve 2026 Hackathon

> **Bridging the gap between AI generation and platform retention.**

## 📊 TL;DR

We analyzed 226.6K behavioral events from 1,472 users to identify what separates power users from churners. Using 51 leakage-free features from the first 7 days of activity, we built a calibrated XGBoost model (PR-AUC 0.269) that identifies users 2.56× more likely to retain. We discovered 6 behavioral archetypes; from "Hands-On Builders" (81% retention) to "Casual Browsers" (4%); and prioritized 4 concrete product interventions with measurable uplift estimates.

📄 **[Read the full Written Summary →](builderflow/WRITTEN_SUMMARY.md)** (question, methodology, findings)

## 🏗️ Project Structure

```
builderflow/
├── canvas.yaml                    # Canvas definition & DAG
├── Development/                   # Main analysis layer (20 blocks)
│   ├── project_config_setup.py         # Config & data loading
│   ├── load_and_prepare_cohort.py      # Cohort filtering
│   ├── feature_engineering_7d_window.py # 51 features from 7-day window
│   ├── feature_schema_and_heatmap.py   # Schema validation & correlation
│   ├── kmeans_archetype_clustering.py  # 6 behavioral archetypes
│   ├── xgboost_bayesian_opt_model.py   # XGBoost + Bayesian HPO
│   ├── shap_advanced_analysis.py       # SHAP interpretability
│   ├── uplift_intervention_scoring.py  # 4 interventions, priority ranking
│   ├── executive_summary_charts.py     # Executive dashboard
│   ├── executive_narrative.md          # Written report
│   └── ... (EDA, baseline models, calibration, propensity analysis)
├── ScheduledJob/                  # Deployed scoring pipeline
│   ├── retention_scoring_job.py        # Daily user risk scoring
│   └── layer.yaml                      # Schedule configuration
└── REPRODUCTION_GUIDE.md           # Step-by-step reproduction guide
```

## 🔄 Pipeline Architecture

```mermaid
flowchart TD
    A["📂 project_config_setup"] --> B["🔄 load_and_prepare_cohort"]
    B --> C["⚙️ feature_engineering_7d_window"]
    C --> D["📊 feature_schema_and_heatmap"]

    D --> E1["📈 eda_event_taxonomy"]
    D --> E2["📈 eda_user_timelines"]
    D --> E3["📈 eda_retention_by_behavior"]

    D --> F["🧩 kmeans_archetype_clustering"]
    D --> G["🧮 compute_labels_and_features"]

    F --> H["🌲 xgboost_bayesian_opt_model"]
    G --> I["📏 train_baseline_and_main_models"]
    G --> J["🧩 behavioral_clustering"]
    G --> K["🔬 propensity_impact_analysis"]

    H --> L["🔍 shap_advanced_analysis"]
    H --> M["🎯 uplift_intervention_scoring"]
    F --> M

    I --> N["📐 calibration_and_comparison_charts"]
    I --> O["🔍 shap_analysis"]
    O --> P["📊 executive_summary_charts"]
    J --> P
    I --> P
    I --> K

    M --> Q["📁 scored_user_table.csv"]

    subgraph Scheduled["⏰ ScheduledJob Layer"]
        R["🔄 retention_scoring_job"]
        R --> S["📁 scored_users_latest.csv"]
    end

    H -.->|"model + calibrator"| R
    F -.->|"archetypes"| R

    style A fill:#2d2d30,stroke:#A1C9F4,color:#fbfbff
    style C fill:#2d2d30,stroke:#8DE5A1,color:#fbfbff
    style F fill:#2d2d30,stroke:#D0BBFF,color:#fbfbff
    style H fill:#2d2d30,stroke:#FFB482,color:#fbfbff
    style M fill:#2d2d30,stroke:#ffd400,color:#fbfbff
    style R fill:#2d2d30,stroke:#17b26a,color:#fbfbff
    style Scheduled fill:#1a1a1d,stroke:#17b26a,color:#fbfbff
```

## 🔑 Key Results

| Metric | Value |
|--------|-------|
| PR-AUC (XGB Calibrated) | **0.269** |
| Lift @ Top 10% | **2.56×** |
| Rolling CV PR-AUC | 0.249 ± 0.058 |
| Brier Score (calibrated) | **0.013** |
| Archetypes | **6** (stable, silhouette=0.52) |
| Top Driver | `active_days` (SHAP=0.229) |
| #1 Intervention | Agent→Block UI (+21 retained users) |

## 🚀 Deployed Functionality

**Scheduled Job**: Daily retention scoring pipeline that re-scores all users, assigns risk tiers, and outputs CSV for the product team.

## 📝 How to Reproduce

See [REPRODUCTION_GUIDE.md](builderflow/REPRODUCTION_GUIDE.md) for step-by-step instructions.

## 🛠️ Tech Stack

- **Python** (pandas, numpy, scikit-learn, xgboost, matplotlib)
- **Zerve Platform** (Canvas, Code Blocks, Scheduled Jobs)
- **Techniques**: Temporal train/test split, Bayesian HPO, isotonic calibration, SHAP TreeExplainer, K-Means clustering, propensity score analysis, uplift estimation

## 📄 License

MIT License: see [LICENSE](LICENSE)
