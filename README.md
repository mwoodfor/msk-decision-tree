# msk-decision-tree

A regression tree analysis for predicting 12-month musculoskeletal (MSK) risk scores from binary clinical features. Produces a publication-quality decision tree visualisation using `rpart`, `partykit`, and `ggparty`.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Feature Selection](#feature-selection)
  - [Model Fitting](#model-fitting)
  - [Pruning Strategy](#pruning-strategy)
  - [Visualisation](#visualisation)
- [Output](#output)
- [Interpreting the Tree](#interpreting-the-tree)
- [Extending the Analysis](#extending-the-analysis)
- [Contributing](#contributing)

---

## Overview

This script fits a regression decision tree that predicts a continuous 12-month MSK risk score from six binary clinical indicators. The tree is pruned using the **1-SE rule** to favour interpretability and reduce overfitting. Terminal leaf nodes display both a risk-score histogram and the segment mean (μ), making the output immediately accessible to clinical and non-technical stakeholders.

---

## Repository Structure

```
msk-decision-tree/
├── msk_decision_tree.R          # Main analysis script
├── README.md                    # This file
└── outputs/
    └── decision_tree_msk_ggparty.png   # Generated on run
```

---

## Prerequisites

**R version:** ≥ 4.1.0 recommended

Install required packages from CRAN:

```r
install.packages(c("partykit", "ggparty", "ggplot2", "scales", "patchwork", "rpart", "dplyr"))
```

| Package | Version tested | Purpose |
|---|---|---|
| `rpart` | 4.1.x | Fits the regression tree |
| `partykit` | 1.2.x | Tree object representation |
| `ggparty` | 1.0.x | ggplot2 extension for tree visualisation |
| `ggplot2` | 3.4.x | Core plotting |
| `scales` | 1.2.x | Axis label formatting |
| `patchwork` | 1.1.x | Multi-panel plot composition |
| `dplyr` | 1.1.x | Data wrangling (`%>%`, `select`) |

---

## Data Requirements

The script expects a data frame named **`df_raw`** to be present in the R environment before running. It must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `risk_score_12m` | `numeric` | **Outcome.** Continuous MSK risk score at 12 months |
| `has_pain_general` | `integer` / `logical` | Binary flag: any general pain diagnosis |
| `has_conservative_care` | `integer` / `logical` | Binary flag: conservative care utilisation |
| `has_pt` | `integer` / `logical` | Binary flag: physical therapy utilisation |
| `has_inj` | `integer` / `logical` | Binary flag: injection utilisation |
| `has_lifestyle_comorbidity` | `integer` / `logical` | Binary flag: lifestyle-related comorbidity present |
| `has_acute_care` | `integer` / `logical` | Binary flag: acute care utilisation |

Binary columns should be encoded as `0`/`1` integers or `TRUE`/`FALSE` logicals. Additional columns in `df_raw` are ignored.

> **Minimum recommended rows:** ~500 (to support `minsplit = 30` at multiple levels of the tree without degenerate leaves).

---

## Usage

1. Load `df_raw` into your R session.
2. Source the script:

```r
source("msk_decision_tree.R")
```

The script will:
- Select and clean the relevant columns
- Split data 80/20 into training and test sets (`set.seed(42)`)
- Fit and prune the regression tree
- Render and save the visualisation

The output PNG is written to the working directory as `decision_tree_msk_ggparty.png`.

---

## Methodology

### Feature Selection

Six binary clinical features are used as predictors. They represent three clinical domains:

- **Pain burden:** `has_pain_general`
- **Care utilisation:** `has_conservative_care`, `has_pt`, `has_inj`, `has_acute_care`
- **Comorbidity:** `has_lifestyle_comorbidity`

### Model Fitting

The tree is fit using `rpart` with `method = "anova"` (regression mode). Key hyperparameters:

| Parameter | Value | Rationale |
|---|---|---|
| `maxdepth` | `4` | Caps tree depth for interpretability |
| `minsplit` | `30` | Minimum observations required in a node to attempt a split |
| `cp` | `0.005` | Complexity parameter; splits must improve R² by ≥ 0.5% |

### Pruning Strategy

The **1-SE rule** is applied after fitting:

1. Identify the tree size with minimum cross-validated error (`xerror`).
2. Compute a threshold: `min(xerror) + 1 × xstd`.
3. Select the **smallest** tree whose `xerror` falls within that threshold.

This produces a more conservative model than simply minimising cross-validated error alone, trading a small amount of predictive accuracy for a simpler, more generalisable tree.

### Visualisation

The tree is converted to a `partykit` party object and rendered with `ggparty`. Each terminal leaf contains:

- A **mini histogram** of `risk_score_12m` within that patient segment, filled with a teal-to-coral gradient representing low-to-high risk.
- A **mean label** (`μ = X.XX`) positioned below the histogram.

Inner nodes display the splitting variable name; edges display the Yes/No branch condition.

---

## Output

`decision_tree_msk_ggparty.png` — 14 × 9 inches at 300 dpi.

Example layout:

```
              [Split Variable A]
             /                  \
     [Split Variable B]      [Split Variable C]
      /          \               /          \
  [Leaf 1]   [Leaf 2]      [Leaf 3]      [Leaf 4]
  μ = 0.23   μ = 0.51      μ = 0.44      μ = 0.78
```

Each leaf shows a histogram and mean risk score for the patient segment that reaches that node.

---

## Interpreting the Tree

- **Root node split:** The first split uses the single feature that most reduces variance in `risk_score_12m` across the training set. This is the strongest univariate predictor.
- **Leaf mean (μ):** The average predicted risk score for all patients in that segment. Higher values indicate greater predicted MSK burden at 12 months.
- **Leaf histogram:** Shows the distribution shape within the segment. Wide or right-skewed distributions indicate heterogeneous risk even within that segment.
- **Depth of a split:** Features appearing closer to the root explain more variance overall. Features appearing only in deeper branches have conditional importance.

---

## Extending the Analysis

**Add test-set metrics:**
```r
preds <- predict(tree_pruned, newdata = test)
rmse  <- sqrt(mean((test$risk_score_12m - preds)^2))
cat("Test RMSE:", round(rmse, 3), "\n")
```

**Add more predictors:** Expand the `select()` call in the data preparation section and include any new column in the model formula (or use `. ~ .` to include all columns automatically).

**Try a deeper tree:** Increase `maxdepth` from `4` to `5` or `6`, but re-check the pruned depth — the 1-SE rule may still return a shallow tree.

**Export to PDF:**
```r
ggsave("decision_tree_msk_ggparty.pdf", plot = p, width = 14, height = 9)
```

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

---

*Analysis authored by [@mwoodfor](https://github.com/mwoodfor)*
