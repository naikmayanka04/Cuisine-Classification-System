[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/naikmayanka04/Cuisine-Classification-System)
# Restaurant Cuisine Classification

Predicting a restaurant's primary cuisine from structured attributes (location, price,
ratings, and service options) using classical machine learning.

## Project Overview

| | |
|---|---|
| **Task** | Multi-class classification |
| **Target** | Restaurant's primary cuisine (13 classes: 12 most common cuisines + "Other") |
| **Data** | 9,551 restaurants, 21 columns (Zomato-style restaurant listing dataset) |
| **Models** | Random Forest (primary), Logistic Regression (baseline) |
| **Best result** | Random Forest — 0.27 accuracy / 0.26 macro F1 / 0.27 weighted F1 |

This repo is intentionally built around an honest finding rather than an inflated one: with
only structured/tabular features available (no restaurant name text, no menu, no reviews),
cuisine is genuinely hard to predict for many restaurants. Section
["Challenges & Biases"](#challenges--biases) below explains why, in detail — that analysis
is as much a part of the deliverable as the model itself.

## Repository Structure

```
cuisine-classification/
├── data/
│   └── Restaurant_Dataset.csv        # source dataset (converted from the provided .xlsx)
├── notebooks/
│   └── cuisine_classification.ipynb  # full walkthrough: EDA -> preprocessing -> training -> evaluation -> analysis
├── src/
│   ├── preprocessing.py              # data cleaning, feature engineering, encoding
│   └── train.py                      # CLI training/evaluation pipeline
├── outputs/
│   ├── classification_report.txt     # precision/recall/F1 per class
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── per_class_performance.csv     # per-cuisine metrics, sorted by F1
│   └── model.joblib                  # trained model + scaler + feature columns
├── requirements.txt
└── README.md
```

## How to Run

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Train + evaluate (writes everything in outputs/)
python src/train.py --data data/Restaurant_Dataset.csv --model random_forest

# Or try the logistic regression baseline
python src/train.py --data data/Restaurant_Dataset.csv --model logistic_regression
```

Or open `notebooks/cuisine_classification.ipynb` for the full narrated walkthrough with
plots and commentary (outputs are pre-rendered in the notebook, so it can be read directly
on GitHub without running anything).

## Methodology

### 1. Framing the label

Restaurants in the raw data list multiple cuisines as a comma-separated string
(e.g. `"North Indian, Chinese"`). Rather than a multi-label formulation, this project
classifies each restaurant's **primary (first-listed) cuisine** — the standard simplification
for this dataset. The raw primary-cuisine label space has **119 distinct values**, the vast
majority with only a handful of examples — not enough for any model to learn or for a metric
to meaningfully evaluate. The **12 most frequent** primary cuisines are kept as explicit
classes; everything else is grouped into an **`Other`** class. This choice trades some label
granularity for classes that are actually learnable and evaluable, and is documented here so
it can be revisited (e.g. by increasing `top_n` in `preprocessing.py`, or reframing as
multi-label — see [Future Work](#future-work)).

### 2. Handling missing values

- 9 rows with no `Cuisines` value are dropped (no label can be derived for them).
- `Average Cost for two` has a number of `0` entries that aren't real prices; these are
  treated as missing and imputed with the **median cost within the restaurant's country**
  (prices aren't comparable across currencies/economies).
- Remaining numeric fields (`Votes`, `Aggregate rating`, `Longitude`, `Latitude`) are imputed
  with the column median.
- Categorical fields (`City`, `Currency`, `Rating text`) are imputed with `"Unknown"`.

### 3. Feature engineering

- `Cuisine Count` — the number of cuisines a restaurant lists — is kept as a feature, since
  this signal would otherwise be lost once the target is reduced to a single primary cuisine.
- Long-tail `City` values (141 distinct cities, heavily dominated by Delhi-NCR) are bucketed:
  the top 20 cities are kept explicitly, the rest become `"Other City"`, for the same reason
  the cuisine long tail is bucketed.

### 4. Encoding

- Yes/No service columns (`Has Table booking`, `Has Online delivery`, etc.) are mapped to 0/1.
- `City` and `Currency` are one-hot encoded.
- All numeric features are standardized (`StandardScaler`) before training.

### 5. Train/test split

An 80/20 split, **stratified on the target class**, so every cuisine (including the smaller
ones) is proportionally represented in both sets.

### 6. Models

- **Logistic Regression** (`class_weight='balanced'`) as an interpretable linear baseline.
- **Random Forest** (`class_weight='balanced_subsample'`) as the primary model — better suited
  to the mixed numeric/categorical feature set and to feature interactions (e.g. price range
  × city).

Both are weighted to counter class imbalance rather than let the model default toward the
majority class — see below.

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 0.15 | 0.15 | 0.14 |
| **Random Forest** | **0.27** | **0.26** | **0.27** |

Full precision/recall/F1 per class: [`outputs/classification_report.txt`](outputs/classification_report.txt)
and [`outputs/per_class_performance.csv`](outputs/per_class_performance.csv).

Random Forest outperforms Logistic Regression substantially, consistent with the feature set
having non-linear structure (interactions between price, location, and service options) that
a linear model can't capture.

## Challenges & Biases

**Class imbalance dominates the majority-class baseline.** `North Indian` alone accounts for
~31% of restaurants (a reflection of the dataset being dominated by Delhi-NCR listings). An
unweighted model can reach a *higher raw accuracy* simply by defaulting ambiguous cases to
`North Indian`, while doing poorly on every minority cuisine. We use `class_weight='balanced_subsample'`
specifically to counter this, at some cost to raw accuracy — the gap between accuracy and
macro F1 in the results table is a direct measurement of this trade-off.

**Feature ceiling.** Two similarly priced, similarly rated restaurants in the same city could
easily be a Chinese takeout spot or an Italian bistro — nothing in `price range`, `votes`,
`rating`, or `location` distinguishes them. These features describe a restaurant's *quality
and market positioning*, not its *cuisine*. Cuisines that don't have a distinctive
price/location profile in this dataset (`South Indian`, `Chinese`, `Italian`, `Bakery`) are
the hardest for the model, while cuisines with a distinctive profile (`American`, `Pizza`,
the majority `North Indian`) are comparatively easier. This is a genuine data limitation, not
just a modeling one — see [Future Work](#future-work) for the fix.

**Geographic concentration.** ~85% of restaurants are in the Delhi-NCR region (New Delhi,
Gurgaon, Noida, Faridabad). Whatever the model has learned about what predicts, say,
`Continental` or `Italian` cuisine is learned almost entirely from Indian metro dining
patterns, and may not generalize well to the smaller number of restaurants from other
countries/cities present in the data.

**Label-granularity trade-off.** Collapsing 119 cuisines to 12 + `Other` makes the classes
learnable, but it also means the model was never asked to distinguish, e.g., regional Indian
sub-cuisines from one another — they're folded into `Other`. That's a deliberate scope
decision, not a hidden limitation.

## Future Work

- **Add text features.** Restaurant name (`Restaurant Name`) likely carries strong,
  currently-unused cuisine signal (e.g. "Sushi House", "Trattoria..."). A TF-IDF or n-gram
  representation of the name is the single most promising next step for improving accuracy.
- **Multi-label reframing.** Predict the full set of listed cuisines per restaurant instead
  of only the primary one, using a multi-label classifier (e.g. `OneVsRestClassifier`) and
  multi-label metrics (Hamming loss, subset accuracy).
- **Hyperparameter tuning** via cross-validated grid/random search, particularly for the
  Random Forest's depth/leaf-size trade-off between accuracy and generalization.
- **Gradient boosting** (e.g. XGBoost/LightGBM) as a stronger non-linear baseline than Random
  Forest.
- **Rebalance via data collection**, not just class weighting — targeted collection of
  underrepresented cuisines (or non-Delhi-NCR cities) would address the imbalance at its
  source rather than compensating for it algorithmically.

## Data Source

`data/Restaurant_Dataset.csv` — a Zomato-style
restaurant listing dataset covering 9,551 restaurants across 15 countries, with columns for
location, cuisines served, pricing, service options (table booking, online delivery), and
aggregate customer ratings.
