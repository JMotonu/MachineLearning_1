
# ============================================================
# WEEK 2: EXPLORATORY DATA ANALYSIS
# Synthetic Blueberry Yield Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# 1. PROJECT PATHS AND DATA LOADING
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent

data_path = project_folder / "synthetic_blueberry_data_cleaned.csv"

output_folder = project_folder / "EDA_Outputs"
output_folder.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

print("\n========== DATASET OVERVIEW ==========")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

# ------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n========== DESCRIPTIVE STATISTICS ==========")

descriptive_stats = df.describe().T

descriptive_stats["median"] = df.median(numeric_only=True)
descriptive_stats["skewness"] = df.skew(numeric_only=True)

print(descriptive_stats)

descriptive_stats.to_csv(
    output_folder / "descriptive_statistics.csv"
)

# ------------------------------------------------------------
# 3. FREQUENCY DISTRIBUTION OF CLONE SIZE
# ------------------------------------------------------------

print("\n========== CLONE SIZE DISTRIBUTION ==========")

clone_distribution = df["clonesize"].value_counts().sort_index()

print(clone_distribution)

clone_distribution.to_csv(
    output_folder / "clone_size_distribution.csv"
)

# ------------------------------------------------------------
# 4. CHART 1: DISTRIBUTIONS OF ALL VARIABLES
# ------------------------------------------------------------

columns = df.columns.tolist()

number_of_columns = 4
number_of_rows = int(np.ceil(len(columns) / number_of_columns))

fig, axes = plt.subplots(
    number_of_rows,
    number_of_columns,
    figsize=(17, number_of_rows * 3.5)
)

axes = axes.flatten()

for i, column in enumerate(columns):

    axes[i].hist(
        df[column],
        bins=25,
        edgecolor="black"
    )

    axes[i].set_title(column)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

for j in range(len(columns), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Distribution of Synthetic Blueberry Dataset Variables",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    output_folder / "01_variable_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 5. CHART 2: CORRELATION HEATMAP
# ------------------------------------------------------------

correlation_matrix = df.corr(numeric_only=True)

correlation_matrix.to_csv(
    output_folder / "correlation_matrix.csv"
)

fig, ax = plt.subplots(figsize=(14, 11))

image = ax.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))

ax.set_xticklabels(
    correlation_matrix.columns,
    rotation=90
)

ax.set_yticklabels(correlation_matrix.columns)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):

        ax.text(
            j, i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=7
        )

plt.colorbar(image, ax=ax, label="Correlation coefficient")

ax.set_title(
    "Correlation Matrix of Synthetic Blueberry Variables",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    output_folder / "02_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 6. CHART 3: YIELD AGAINST FRUIT CHARACTERISTICS
# ------------------------------------------------------------

fruit_variables = ["fruitset", "fruitmass", "seeds"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for i, column in enumerate(fruit_variables):

    axes[i].scatter(
        df[column],
        df["yield"],
        alpha=0.6
    )

    axes[i].set_title(f"Yield vs {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Yield")

fig.suptitle(
    "Relationship Between Yield and Fruit Characteristics",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    output_folder / "03_yield_fruit_relationships.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 7. CHART 4: YIELD AGAINST POLLINATOR VARIABLES
# ------------------------------------------------------------

pollinator_variables = [
    "honeybee",
    "bumbles",
    "andrena",
    "osmia"
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

axes = axes.flatten()

for i, column in enumerate(pollinator_variables):

    axes[i].scatter(
        df[column],
        df["yield"],
        alpha=0.6
    )

    axes[i].set_title(f"Yield vs {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Yield")

fig.suptitle(
    "Relationship Between Yield and Pollinator Variables",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    output_folder / "04_yield_pollinator_relationships.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 8. CHART 5: YIELD BY CLONE SIZE
# ------------------------------------------------------------

clone_sizes = sorted(df["clonesize"].unique())

yield_by_clone = [
    df.loc[df["clonesize"] == size, "yield"].values
    for size in clone_sizes
]

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(
    yield_by_clone,
    tick_labels=[str(size) for size in clone_sizes]
)

ax.set_title(
    "Distribution of Blueberry Yield by Clone Size",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Clone Size")
ax.set_ylabel("Yield")

plt.tight_layout()

plt.savefig(
    output_folder / "05_yield_by_clone_size.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 9. CHART 6: YIELD AGAINST RAINFALL DAYS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    df["RainingDays"],
    df["yield"],
    alpha=0.6
)

ax.set_title(
    "Relationship Between Rainfall Days and Blueberry Yield",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Raining Days")
ax.set_ylabel("Yield")

plt.tight_layout()

plt.savefig(
    output_folder / "06_yield_vs_raining_days.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 10. YIELD CORRELATION RANKING
# ------------------------------------------------------------

yield_correlations = (
    correlation_matrix["yield"]
    .drop("yield")
    .sort_values(key=abs, ascending=False)
)

print("\n========== CORRELATION WITH YIELD ==========")
print(yield_correlations)

yield_correlations.to_csv(
    output_folder / "yield_correlations.csv"
)

# ------------------------------------------------------------
# 11. SAVE EDA SUMMARY
# ------------------------------------------------------------

print("\n========== EDA COMPLETED ==========")

print("Records analysed:", len(df))
print("Variables analysed:", len(df.columns))

print("\nOutput folder:")
print(output_folder)

print("\nGenerated files:")
for file in sorted(output_folder.iterdir()):
    print(file.name)

plt.savefig(
    output_folder / "01_variable_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()