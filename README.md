# Synthetic Blueberry Yield Data Generation and Analysis

## 1. Project Overview

This project explores the generation, cleaning, validation and exploratory analysis of synthetic blueberry yield data using a fine-tuned GPT-2 Medium model.

The project uses the Generative Recurrent Transformer (GReaT) approach, implemented through the `be-great` framework, to generate synthetic tabular observations based on a blueberry yield dataset.

The current phase focuses on generating 1,000 synthetic observations, assessing their structural and domain-level validity, and conducting initial exploratory data analysis (EDA).

## 2. Project Objectives

The project aims to:

1. Generate synthetic blueberry yield data using a fine-tuned GPT-2 Medium model.
2. Clean and prepare the generated dataset for analysis.
3. Assess the quality and validity of generated observations using data-quality and domain-specific checks.
4. Explore statistical distributions, variable relationships and patterns associated with blueberry yield.
5. Prepare the synthetic dataset for further evaluation and potential machine learning applications.

## 3. Technology Stack

The project uses the following tools and technologies:

- **Python 3.12.8** – primary programming language.
- **PyTorch** – model loading and inference.
- **Hugging Face Transformers** – GPT-2 model architecture and tokenizer.
- **GReaT / be-great** – synthetic tabular data generation.
- **Pandas** – data manipulation and analysis.
- **NumPy** – numerical operations.
- **Matplotlib** – data visualisation.
- **VS Code** – development environment.
- **Kaggle** – source of the fine-tuned blueberry yield model.

## 4. Dataset Description

The project generated 1,000 synthetic observations containing 18 columns in the original generated file.

The variables describe:

| Category | Variables |
|---|---|
| Clone characteristics | `clonesize` |
| Pollinator variables | `honeybee`, `bumbles`, `andrena`, `osmia` |
| Temperature variables | `MaxOfUpperTRange`, `MinOfUpperTRange`, `AverageOfUpperTRange`, `MaxOfLowerTRange`, `MinOfLowerTRange`, `AverageOfLowerTRange` |
| Rainfall variables | `RainingDays`, `AverageRainingDays` |
| Fruit characteristics | `fruitset`, `fruitmass`, `seeds` |
| Target variable | `yield` |

The original generated dataset included a `Row#` identifier, which was removed during cleaning. The resulting analytical dataset contains 17 variables.

## 5. Development Workflow and Completed Activities

### 5.1 Model Loading and Synthetic Data Generation

The fine-tuned GPT-2 Medium checkpoint was obtained from Kaggle.

The model was loaded using the GPT-2 architecture and its associated tokenizer. The GReaT framework was configured to generate synthetic tabular observations.

A total of 1,000 synthetic records were generated and saved as:

`synthetic_blueberry_data.csv`

### 5.2 Data Cleaning

The first cleaning stage examined:

- Dataset dimensions and column structure.
- Data types.
- Missing values.
- Duplicate observations.
- Infinite values.
- Negative values.
- Potential outliers using the Interquartile Range (IQR) method.

**Initial cleaning results:**

| Quality check | Result |
|---|---:|
| Original observations | 1,000 |
| Original columns | 18 |
| Missing values | 0 |
| Duplicate substantive records | 0 |
| Infinite values | 0 |
| Negative values | 0 |
| Records retained | 1,000 |
| Final analytical columns | 17 |

Potential outliers were identified but not automatically removed. This preserves observations for further investigation.

The cleaned dataset was saved as:

`synthetic_blueberry_data_cleaned.csv`

A data quality report was also generated:

`data_quality_report.csv`

### 5.3 Domain-Specific Validation

Domain validation was conducted to assess whether generated observations satisfied selected logical and variable-specific constraints.

The checks included:

- Consistency between minimum, maximum and average temperature measurements.
- Validity of clone-size values.
- Rainfall-day constraints.
- Proportion constraints for `AverageRainingDays` and `fruitset`.
- Non-negative pollinator, fruit and yield measurements.

**Validation results:**

| Validation | Result |
|---|---:|
| Clone-size values | 1,000 valid |
| Rainfall-day values | 997 valid |
| Average rainfall | 1,000 valid |
| Fruit set | 1,000 valid |
| Pollinator non-negativity | All 1,000 passed |
| Other non-negative variables | All 1,000 passed |
| Temperature consistency | All 1,000 passed |

Three observations contained a fractional `RainingDays` value of 3.77. These were flagged for further investigation rather than automatically corrected or deleted.

### 5.4 Exploratory Data Analysis (In Progress)

Initial exploratory data analysis has commenced using Pandas, NumPy and Matplotlib.

The analysis includes:

- Descriptive statistics.
- Frequency distribution of clone sizes.
- Distribution plots for individual variables.
- Correlation analysis.
- Relationships between yield and fruit characteristics.
- Relationships between yield and pollinator variables.
- Yield comparisons across clone sizes.
- Relationship between rainfall days and yield.

Initial descriptive statistics and clone-size distributions have been generated.

The clone-size distribution shows that 12.5 and 25 account for most generated observations. Further analysis will investigate whether this concentration reflects the distribution of the original training data.

## 6. Project Files

The current project structure is organised as follows:

```text
Week_2/
│
├── test_blueberry_model.py
├── 01_data_cleaning.py
├── 02_domain_validation.py
├── 03_inspect_suspicious_values.py
├── 04_corrected_domain_validation.py
├── 05_exploratory_data_analysis.py
│
├── synthetic_blueberry_data.csv
├── synthetic_blueberry_data_cleaned.csv
├── data_quality_report.csv
│
├── EDA_Outputs/
│   ├── 01_variable_distributions.png
│   ├── clone_size_distribution.csv
│   └── descriptive_statistics.csv
│
├── README.md
└── .gitignore
```

Additional EDA charts and statistical outputs will be added as the analysis progresses.

## 7. Running the Project

The project uses a dedicated Python 3.12 virtual environment named `.venv312`.

Run the scripts from the `Week_2` project directory using:

**Data cleaning**

```bash
./.venv312/bin/python 01_data_cleaning.py
```

**Domain validation**

```bash
./.venv312/bin/python 02_domain_validation.py
```

**Corrected domain validation**

```bash
./.venv312/bin/python 04_corrected_domain_validation.py
```

**Exploratory data analysis**

```bash
./.venv312/bin/python 05_exploratory_data_analysis.py
```

## 8. Current Project Status

| Project stage | Status |
|---|---|
| Environment setup | Completed |
| Fine-tuned model loading | Completed |
| Synthetic data generation | Completed |
| Initial data cleaning | Completed |
| Domain-specific validation | Completed; 3 observations flagged |
| Exploratory data analysis | In progress |
| Real-versus-synthetic comparison | Pending |
| Statistical fidelity assessment | Pending |
| Further modelling and evaluation | Pending |


## 9. Reproducibility and Data Integrity

The original generated dataset is retained separately from the cleaned analytical dataset.

Potentially unusual observations are flagged for investigation rather than automatically removed. This supports transparency, reproducibility and traceability throughout the data preparation process.

The current validation results should not be interpreted as proof of statistical equivalence between the synthetic and original datasets. That assessment will be conducted in a subsequent project phase.
