import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent

file_path = project_folder / "synthetic_blueberry_data_cleaned.csv"

df = pd.read_csv(file_path)

print("\n========== CLONESIZE INSPECTION ==========")

print("\nUnique clonesize values:")
print(sorted(df["clonesize"].unique()))

print("\nMost frequent clonesize values:")
print(df["clonesize"].value_counts().head(20))

print("\nNon-integer clonesize records:")
print(
    df.loc[
        ~df["clonesize"].mod(1).eq(0),
        ["clonesize", "yield"]
    ].head(20)
)

print("\n========== RAINING DAYS INSPECTION ==========")

print("\nUnique RainingDays values:")
print(sorted(df["RainingDays"].unique()))

print("\nNon-integer RainingDays records:")
print(
    df.loc[
        ~df["RainingDays"].mod(1).eq(0),
        ["RainingDays", "AverageRainingDays", "yield"]
    ]
)

print("\n========== INSPECTION COMPLETED ==========")