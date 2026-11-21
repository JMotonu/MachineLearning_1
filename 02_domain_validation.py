
# ============================================================
# BLUEBERRY YIELD DATA - DOMAIN VALIDATION
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load cleaned dataset

project_folder = Path(__file__).resolve().parent

file_path = project_folder / "synthetic_blueberry_data_cleaned.csv"

df = pd.read_csv(file_path)

print("\n========== DOMAIN VALIDATION ==========")

# 2. Temperature range consistency

print("\n--- Temperature Range Checks ---")

temperature_checks = {
    "Upper average within range": (
        (df["AverageOfUpperTRange"] >= df["MinOfUpperTRange"]) &
        (df["AverageOfUpperTRange"] <= df["MaxOfUpperTRange"])
    ),

    "Lower average within range": (
        (df["AverageOfLowerTRange"] >= df["MinOfLowerTRange"]) &
        (df["AverageOfLowerTRange"] <= df["MaxOfLowerTRange"])
    ),

    "Upper maximum >= upper minimum": (
        df["MaxOfUpperTRange"] >= df["MinOfUpperTRange"]
    ),

    "Lower maximum >= lower minimum": (
        df["MaxOfLowerTRange"] >= df["MinOfLowerTRange"]
    )
}

for name, result in temperature_checks.items():

    print(f"{name}: {result.sum()} / {len(df)} passed")

    if (~result).sum() > 0:
        print("Violations:", (~result).sum())


# 3. Proportion-like variable checks

print("\n--- Proportion Checks ---")

proportion_columns = [
    "AverageRainingDays",
    "fruitset",
    "honeybee",
    "bumbles",
    "andrena",
    "osmia"
]

for column in proportion_columns:

    valid = df[column].between(0, 1)

    print(
        f"{column}: {valid.sum()} / {len(df)} "
        "within [0, 1]"
    )


# 4. Non-negative agricultural variables

print("\n--- Non-negative Variable Checks ---")

nonnegative_columns = [
    "clonesize",
    "RainingDays",
    "fruitmass",
    "seeds",
    "yield"
]

for column in nonnegative_columns:

    valid = df[column] >= 0

    print(
        f"{column}: {valid.sum()} / {len(df)} "
        "non-negative"
    )


# 5. Integer-like variable checks

print("\n--- Integer-like Variable Checks ---")

integer_columns = [
    "clonesize",
    "RainingDays"
]

for column in integer_columns:

    integer_like = np.isclose(
        df[column],
        np.round(df[column])
    )

    print(
        f"{column}: {integer_like.sum()} / {len(df)} "
        "integer-like"
    )


# 6. Overall summary

print("\n========== VALIDATION COMPLETED ==========")

print("Records examined:", len(df))

print("Variables examined:", len(df.columns))
