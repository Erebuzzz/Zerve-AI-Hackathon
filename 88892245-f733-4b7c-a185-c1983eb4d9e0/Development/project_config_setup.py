
# ─────────────────────────────────────────────────────
# CONFIG — Project Constants
# ─────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Cohort cutoff: only users whose first event is on or before this date
COHORT_CUTOFF = pd.Timestamp("2025-11-01")

# Early-window: number of days after first event to define "early behaviour"
EARLY_WINDOW_DAYS = 7

# Label horizons (in days after first event) for retention / upgrade labels
LABEL_HORIZONS = [7, 14, 30, 60]

# Temporal train/val/test split boundaries (by user first-event date)
TEMPORAL_SPLITS = {
    "train_end": pd.Timestamp("2025-08-01"),
    "val_end":   pd.Timestamp("2025-09-15"),
    # test = everything after val_end up to COHORT_CUTOFF
}

# Mode flags
MODE_DEBUG = False        # If True, sample a small fraction for speed
DEBUG_SAMPLE_FRAC = 0.05  # fraction of users to keep in debug mode

print("✅ CONFIG constants defined")
print(f"   COHORT_CUTOFF      = {COHORT_CUTOFF.date()}")
print(f"   EARLY_WINDOW_DAYS  = {EARLY_WINDOW_DAYS}")
print(f"   LABEL_HORIZONS     = {LABEL_HORIZONS}")
print(f"   TEMPORAL_SPLITS    = { {k: str(v.date()) for k, v in TEMPORAL_SPLITS.items()} }")
print(f"   MODE_DEBUG         = {MODE_DEBUG}")

# ─────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────
_raw = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv", low_memory=False)
print(f"\n📦 Raw dataset: {_raw.shape[0]:,} rows × {_raw.shape[1]} cols")
print(f"   Columns: {list(_raw.columns)}")
print(f"\nDtypes:\n{_raw.dtypes.to_string()}")
print(f"\nFirst 5 rows:\n{_raw.head().to_string()}")
print(f"\nNull counts:\n{_raw.isnull().sum().to_string()}")
