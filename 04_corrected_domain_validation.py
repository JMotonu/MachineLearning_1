
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent
file_path = project_folder / "synthetic_blueberry_data_cleaned.csv"

df = pd.read_csv(file_path)

print("\n========== CORRECTED DOMAIN VALIDATION ==========")

# 1. Clone size: valid published values
valid_clone_sizes = [10, 12.5, 20, 25, 37.5, 40]

invalid_clones = ~df["clonesize"].isin(valid_clone_sizes)

print("\nCLONESIZE")
print("Valid records:", (~invalid_clones).sum())
print("Invalid records:", invalid_clones.sum())
print("Observed distribution:")
print(df["clonesize"].value_counts().sort_index())

# 2. RainingDays: whole numbers between 1 and 34
valid_rain = (
    df["RainingDays"].between(1, 34)
    & (df["RainingDays"] % 1 == 0)
)

print("\nRAINING DAYS")
print("Valid records:", valid_rain.sum())
print("Suspicious records:", (~valid_rain).sum())

print("\nSuspicious rainfall observations:")
print(
    df.loc[
        ~valid_rain,
        ["RainingDays", "AverageRainingDays", "yield"]
    ]
)

# 3. AverageRainingDays and fruitset
for column in ["AverageRainingDays", "fruitset"]:
    valid = df[column].between(0, 1)
    print(f"\n{column}: {valid.sum()} / {len(df)} passed")

# 4. Pollinator density: non-negative
pollinator_columns = [
    "honeybee", "bumbles", "andrena", "osmia"
]

for column in pollinator_columns:
    valid = df[column] >= 0
    print(f"{column}: {valid.sum()} / {len(df)} non-negative")

# 5. Other non-negative variables
other_columns = ["fruitmass", "seeds", "yield"]

for column in other_columns:
    valid = df[column] >= 0
    print(f"{column}: {valid.sum()} / {len(df)} non-negative")

print("\n========== VALIDATION COMPLETED ==========")
