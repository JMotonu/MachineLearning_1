
# ============================================================
# BLUEBERRY YIELD SYNTHETIC DATA - DATA CLEANING
# Project: GPT-2 Medium Synthetic Data Generation
# ============================================================

# 1. Import libraries

import pandas as pd
import numpy as np
from pathlib import Path


# 2. Define project file locations

project_folder = Path(__file__).resolve().parent

input_file = project_folder / "synthetic_blueberry_data.csv"

output_file = project_folder / "synthetic_blueberry_data_cleaned.csv"

audit_file = project_folder / "data_quality_report.csv"


# 3. Load original synthetic dataset

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(input_file)

print("Dataset successfully loaded.")

print("Original dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())


# 4. Inspect dataset structure

print("\n========== DATASET STRUCTURE ==========")

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()


# 5. Check missing values

print("\n========== MISSING VALUE ANALYSIS ==========")

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
)

missing_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

print(missing_report)

print(
    "\nTotal missing values:",
    df.isnull().sum().sum()
)


# 6. Check duplicate records

print("\n========== DUPLICATE ANALYSIS ==========")

# Row# is treated as a possible record identifier,
# so exclude it when checking substantive duplicates.

columns_for_duplicate_check = [
    col for col in df.columns
    if col != "Row#"
]

duplicate_count = df.duplicated(
    subset=columns_for_duplicate_check
).sum()

print(
    "Duplicate records (excluding Row#):",
    duplicate_count
)


# 7. Check and convert data types

print("\n========== DATA TYPE CLEANING ==========")

for column in df.columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("\nUpdated data types:")
print(df.dtypes)


# 8. Check infinite values

print("\n========== INFINITE VALUE ANALYSIS ==========")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

infinite_counts = np.isinf(
    df[numeric_columns]
).sum()

print(infinite_counts)

# Replace infinite values with missing values.
# Do not silently remove the affected records.

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# 9. Check negative values

print("\n========== NEGATIVE VALUE ANALYSIS ==========")

negative_counts = (
    df[numeric_columns].lt(0).sum()
)

print(negative_counts)

# Negative values are flagged for investigation.
# They are not automatically deleted because
# domain-specific validation is required.


# 10. Check statistical outliers using IQR

print("\n========== OUTLIER ANALYSIS ==========")

outlier_results = []

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ][column]

    outlier_results.append({
        "Variable": column,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": len(outliers)
    })

outlier_report = pd.DataFrame(outlier_results)

print(outlier_report.to_string(index=False))


# 11. Create a working cleaned dataset

print("\n========== DATASET CLEANING ==========")

clean_df = df.copy()

# Remove exact duplicate records based on
# substantive variables, excluding Row#.

clean_df = clean_df.drop_duplicates(
    subset=columns_for_duplicate_check
)

print(
    "Records after duplicate removal:",
    len(clean_df)
)

# Remove Row# from analytical data if it is
# merely a generated record identifier.

if "Row#" in clean_df.columns:

    clean_df = clean_df.drop(
        columns=["Row#"]
    )

    print("Removed Row# identifier column.")


# 12. Final data quality assessment

print("\n========== FINAL DATA QUALITY ==========")

print("Cleaned dataset shape:", clean_df.shape)

print("\nMissing values after cleaning:")
print(clean_df.isnull().sum())

print("\nFinal descriptive statistics:")
print(clean_df.describe().T)


# 13. Save cleaned dataset

clean_df.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved to:")
print(output_file)


# 14. Save quality report

quality_report = pd.DataFrame({
    "Variable": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Negative Values": negative_counts.values
})

quality_report.to_csv(
    audit_file,
    index=False
)

print("\nData quality report saved to:")
print(audit_file)


# 15. Final summary

print("\n========== CLEANING SUMMARY ==========")

print("Original records:", len(df))

print("Cleaned records:", len(clean_df))

print(
    "Records removed:",
    len(df) - len(clean_df)
)

print("Final variables:", len(clean_df.columns))

print("\nDATA CLEANING PROCESS COMPLETED.")
