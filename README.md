# ✈️ Airline Satisfaction Prediction

> **Kaggle Playground Series — Season 6, Episode 10**

A systematic tabular machine-learning project focused on predicting **airline passenger satisfaction** using gradient-boosting models, feature engineering, tabular foundation models, and ensemble optimization.

The project follows an experiment-driven approach: establish strong baselines, optimize them, measure prediction diversity, test advanced models, and build increasingly sophisticated ensembles.

---

## 🏆 Current Status

**Status:** 🚧 Active Competition Project

| Metric                       |           Result |
| ---------------------------- | ---------------: |
| Training samples             |      **699,635** |
| Test samples                 |      **299,844** |
| Input features               |           **22** |
| Target                       |   `satisfaction` |
| Evaluation metric            |      **ROC-AUC** |
| Best GBDT baseline           |     **0.957875** |
| Initial 3-model blend        |     **0.958226** |
| Best recorded validation AUC |  **0.958413791** |
| Best recorded Public LB      |      **0.95859** |
| AutoGluon                    | ⏳ **Next phase** |

> **Current strategy:** Optimize the existing GBDT + TabICL ensemble while testing whether AutoGluon can provide additional prediction diversity.

---

# 🎯 Objective

The objective is to predict whether an airline passenger is satisfied with their flight experience.

The competition evaluates predictions using **ROC-AUC**, meaning the model is judged primarily on how well it ranks satisfied passengers above unsatisfied passengers.

Rather than relying on a single model, this project explores **ensemble diversity** as one of the main sources of improvement.

---

# 📊 Dataset

The dataset contains passenger demographics, travel information, service ratings, and flight-delay information.

### Dataset Dimensions

```text
Train: 699,635 rows × 23 columns
Test : 299,844 rows × 22 columns
```

The training dataset contains the target column:

```text
satisfaction
```

### Target Distribution

| Satisfaction |   Count | Percentage |
| ------------ | ------: | ---------: |
| False        | 389,296 |     55.64% |
| True         | 310,339 |     44.36% |

The target is reasonably balanced, so no aggressive class-balancing strategy was required.

---

# 🔍 Data Quality

Initial data validation found:

* ✅ No duplicate training rows
* ✅ No duplicate IDs
* ✅ 292 missing values in `Arrival Delay in Minutes`
* ✅ Train/test distributions were checked
* ✅ Categorical and numerical features were identified
* ✅ Potential distribution shifts were investigated

### Categorical Features

```text
Gender
Customer Type
Type of Travel
Class
```

### Numerical Features

```text
id
Age
Flight Distance
Inflight wifi service
Departure/Arrival time convenient
Ease of Online booking
Gate location
Food and drink
Online boarding
Seat comfort
Inflight entertainment
On-board service
Leg room service
Baggage handling
Checkin service
Cleanliness
Departure Delay in Minutes
Arrival Delay in Minutes
```

---

# 🧪 Experiment Pipeline

The project is organized into multiple phases.

```text
Data Analysis
     ↓
GBDT Baselines
     ↓
Model Optimization
     ↓
Prediction Blending
     ↓
Feature Engineering
     ↓
TabPFN
     ↓
TabICL
     ↓
Advanced Ensemble
     ↓
AutoGluon
     ↓
Final Submission
```

---

# 📁 Phase 1 — Data Understanding

### Notebook

`Phase 1 Data.ipynb`

The first phase focused on understanding the dataset before modeling.

### Work performed

* Dataset loading
* Shape verification
* Data-type inspection
* Missing-value analysis
* Duplicate detection
* Target distribution analysis
* Numerical/categorical feature identification
* Train/test distribution comparison
* Basic statistical analysis

**Status:** ✅ Complete

---

# 🧹 Phase 1.1 — Data Preparation

### Notebook

`Phase 1.1 (new Data) .ipynb`

Focused on preparing the dataset for reliable model training and validating preprocessing decisions.

**Status:** ✅ Complete

---

# 🏗️ Phase 1.2 — Training Preparation

### Notebook

`Phase 1.2 (new train).ipynb`

Refined the training pipeline and prepared the data for the first generation of machine-learning models.

**Status:** ✅ Complete

---

# 🌳 Phase 1.5 — GBDT Baselines

### Notebook

`Phase 1.5  (GBDTs).ipynb`

Three strong tabular models were selected as the initial baseline:

* LightGBM
* XGBoost
* CatBoost

### Baseline Results

| Model    | Validation ROC-AUC |
| -------- | -----------------: |
| LightGBM |       **0.957875** |
| XGBoost  |       **0.957796** |
| CatBoost |       **0.957698** |

Although the models produced highly correlated predictions, combining them produced a meaningful improvement.

**Status:** ✅ Complete

---

# ⚙️ Phase 2 — Model Optimization

## Phase 2.1 — LightGBM Optimization

### Notebook

`Phase 2.1 (LGB _ Optimization).ipynb`

LightGBM was systematically optimized across parameters including:

* `num_leaves`
* `learning_rate`
* `min_child_samples`
* `subsample`
* `colsample_bytree`
* `reg_alpha`
* `reg_lambda`
* number of estimators
* early stopping

A strong configuration achieved approximately:

```text
Validation AUC ≈ 0.95813
```

This became an important component of the later ensemble.

**Status:** ✅ Complete

---

## Phase 2.2 — XGBoost Optimization

### Notebook

`Phase 2.2 (XGBoost Optimization).ipynb`

XGBoost was optimized and evaluated for both standalone performance and ensemble contribution.

The focus shifted from:

> "Which model has the highest AUC?"

to:

> "Which model adds the most useful information to the ensemble?"

**Status:** ✅ Complete

---

# 🧬 Phase 3 — Prediction Blending

### Notebook

`Phase 3 (blend).ipynb`

The project then moved from individual model optimization toward **ensemble construction**.

The main models were:

```text
LightGBM
XGBoost
CatBoost
```

The initial ensemble achieved:

```text
ROC-AUC ≈ 0.958226
```

This confirmed that prediction blending could outperform individual models.

### Key principle

A model does not necessarily need to be the strongest standalone model to be valuable.

If its predictions are sufficiently different from the existing champion, it may improve the final ensemble.

**Status:** ✅ Complete

---

# 🧠 Phase 4 — Feature Engineering

### Notebook

`Phase 4 (Feature Engineering).ipynb`

Multiple feature-engineering strategies were investigated.

The goal was not simply to create more features, but to determine whether new representations contained information that the existing GBDT models were missing.

Experiments included:

* service-related feature combinations
* interaction features
* segment-based features
* specialized feature subsets
* alternative representations

Not every experiment improved the score.

For example, one segment-interaction experiment produced:

```text
Baseline AUC : 0.957885874
Segment AUC  : 0.957865203
Delta        : -0.000020671
```

This experiment was therefore rejected.

### Philosophy

> Negative experiments are useful results.

Recording unsuccessful approaches prevents repeatedly spending time on strategies that have already been tested.

**Status:** ✅ Complete

---

# 🤖 Phase 5A — TabPFN

### Notebook

`Phase 5A (Tabpfn).ipynb`

TabPFN was evaluated as a tabular foundation-model approach.

The purpose was not simply to replace the existing GBDT models, but to determine whether a fundamentally different modeling approach could provide useful ensemble diversity.

**Status:** ✅ Evaluated

---

# 🧠 Phase 5B — TabICL

### Notebook

`Phase 5B (TablCL).ipynb`

TabICL was introduced as another source of model diversity.

Multiple TabICL configurations were investigated, including:

```text
TabICL 50K
TabICL 100K
```

Rather than allowing TabICL to dominate the ensemble, its predictions were given relatively small weights and evaluated as complementary information.

---

## 🏆 Best TabICL Validation Blend

The strongest recorded validation combination was:

```text
GBDT Champion : 86.0%
TabICL 50K    :  9.1%
TabICL 100K   :  4.9%
```

Result:

```text
Validation AUC = 0.958413791
```

Improvement over the previous tested blend:

```text
+0.000000295
```

The improvement is extremely small, demonstrating an important lesson:

> Tabular foundation models do not automatically outperform strong GBDTs on large structured datasets.

Their value can instead come from **prediction diversity**.

**Status:** ✅ Evaluated

---

# 🏎️ Phase 5C — Full Ensemble

### Notebook

`Phase 5C (Full Blend) .ipynb`

The current advanced ensemble combines several independent prediction sources.

---

## 1️⃣ GBDT Ensemble

The GBDT component is constructed as:

```text
GBDT =
0.75 × (
    0.57 × LightGBM
  + 0.24 × XGBoost
  + 0.19 × CatBoost
)
+ 0.25 × Specialized LightGBM
```

This combines:

* optimized LightGBM
* baseline XGBoost
* baseline CatBoost
* specialized LightGBM

---

## 2️⃣ TabICL Ensemble

The GBDT prediction is then combined with TabICL:

```text
BASE =
0.86 × GBDT
+ 0.091 × TabICL_50K
+ 0.049 × TabICL_100K
```

---

## 3️⃣ Additional CV-LightGBM Component

A separate LightGBM ensemble was introduced to provide another source of prediction diversity.

Two versions were used:

```text
LightGBM — all features
LightGBM — without delay features
```

Both were trained using:

```text
5-Fold Stratified Cross Validation
```

Their predictions were combined using rank averaging.

---

## 4️⃣ Final Rank Blend

The final architecture uses:

```text
FINAL =
0.70 × rank(BASE)
+ 0.30 × rank(CV-LightGBM)
```

This produced the best recorded public leaderboard result:

```text
🏆 Public LB = 0.95859
```

**Status:** ✅ Current recorded leaderboard champion

---

# 📈 Performance Progress

| Stage                      |                Result |
| -------------------------- | --------------------: |
| LightGBM baseline          |              0.957875 |
| XGBoost baseline           |              0.957796 |
| CatBoost baseline          |              0.957698 |
| Initial GBDT blend         |          **0.958226** |
| Optimized GBDT experiments |         **~0.95813+** |
| TabICL-assisted validation |       **0.958413791** |
| Full rank-based ensemble   | **0.95859 Public LB** |

The improvement may appear small numerically, but at a high Kaggle leaderboard level, gains of a few `1e-4` can represent meaningful ranking improvements.

---

# 🚀 Next Phase — AutoGluon

## ⏳ AutoGluon has NOT been run yet.

This is the next planned major experiment.

The goal is to determine whether AutoGluon can discover useful model combinations that our manually constructed ensemble has not captured.

### Planned workflow

```text
Train AutoGluon
       ↓
Generate validation predictions
       ↓
Generate test predictions
       ↓
Measure correlation with champion
       ↓
Evaluate standalone AUC
       ↓
Blend with champion
       ↓
Compare validation AUC
       ↓
Submit if justified
```

### Important rule

AutoGluon will **not automatically replace the current champion**.

It will first be treated as another candidate ensemble component.

The key question is:

> **Does AutoGluon provide genuinely new predictive information?**

If AutoGluon has high correlation with the existing ensemble and provides little or no improvement, it will be rejected.

If it provides meaningful diversity and improves validation performance, it will be incorporated into the final ensemble.

---

# 🧪 Experimental Philosophy

This project follows several principles.

### 1. Strong baselines first

Start with proven tabular algorithms before moving to more experimental models.

### 2. Optimize before expanding

A strong baseline should be optimized before adding unnecessary model complexity.

### 3. Diversity matters

The goal is not to collect as many models as possible.

The goal is to collect models that make **different mistakes**.

### 4. Validation before leaderboard

Public leaderboard improvements are useful, but validation performance is used as the primary experimental signal.

### 5. Reject weak experiments

If a feature set or model does not improve the system, it should not be retained merely because it looks sophisticated.

### 6. Track everything

Each major experiment is stored as a separate notebook.

This makes the project easier to reproduce and prevents repeating failed experiments.

---

# 📂 Repository Structure

```text
AirLine-Prediction/
│
├── Dataset.zip
│
├── Phase 1 Data.ipynb
├── Phase 1.1 (new Data) .ipynb
├── Phase 1.2 (new train).ipynb
│
├── Phase 1.5  (GBDTs).ipynb
│
├── Phase 2.1 (LGB _ Optimization).ipynb
├── Phase 2.2 (XGBoost Optimization).ipynb
│
├── Phase 3 (blend).ipynb
│
├── Phase 4 (Feature Engineering).ipynb
│
├── Phase 5A (Tabpfn).ipynb
├── Phase 5B (TablCL).ipynb
└── Phase 5C (Full Blend) .ipynb
```

The repository primarily contains the experiment notebooks and dataset.

Large prediction artifacts and intermediate model outputs may be stored separately in Kaggle datasets or local experiment directories rather than committed to Git.

---

# 🛠️ Tech Stack

### Programming

* Python
* Jupyter Notebook

### Data Processing

* Pandas
* NumPy
* SciPy

### Machine Learning

* Scikit-learn
* LightGBM
* XGBoost
* CatBoost

### Tabular Foundation Models

* TabPFN
* TabICL

### Competition Platform

* Kaggle

---

# 📊 Evaluation Metric

The competition uses:

```text
ROC-AUC
```

ROC-AUC evaluates how well the model ranks positive examples above negative examples.

Because ranking is important, this project uses:

* probability blending
* rank blending
* cross-validation
* out-of-fold predictions
* prediction correlation
* ensemble weight search

---

# 🔬 Why Ensemble Diversity?

One of the central findings of this project is that **model strength alone is not enough**.

For example:

```text
Model A → AUC 0.9580
Model B → AUC 0.9577
```

Model B may appear worse.

However, if Model B makes different predictions from Model A, combining them can produce:

```text
Ensemble → AUC 0.9583
```

Therefore, this project evaluates both:

```text
Standalone Performance
        +
Prediction Diversity
```

This principle influenced the transition from simple GBDT blending toward TabPFN, TabICL, and eventually the planned AutoGluon experiment.

---

# 🗺️ Roadmap

### Completed

* [x] Dataset inspection
* [x] Data validation
* [x] Missing-value analysis
* [x] Train/test distribution analysis
* [x] LightGBM baseline
* [x] XGBoost baseline
* [x] CatBoost baseline
* [x] Initial ensemble
* [x] LightGBM optimization
* [x] XGBoost optimization
* [x] Feature-engineering experiments
* [x] TabPFN evaluation
* [x] TabICL evaluation
* [x] Ensemble weight optimization
* [x] Rank-based ensemble
* [x] Public leaderboard submission

### In Progress

* [ ] AutoGluon evaluation
* [ ] AutoGluon prediction-diversity analysis
* [ ] AutoGluon + current champion blend
* [ ] Final model selection
* [ ] Final submission

---

# 🏁 Current Champion

The current system is built around:

```text
Optimized GBDT Ensemble
        +
TabICL 50K
        +
TabICL 100K
        +
5-Fold CV LightGBM
        ↓
Rank-Based Final Blend
```

Best recorded result:

```text
Validation:
0.958413791

Public Leaderboard:
0.95859
```

The system is **not frozen yet**.

The next major experiment is AutoGluon.

---

# 📌 Lessons So Far

### What worked

* Gradient-boosted decision trees
* Carefully tuned LightGBM
* Multi-model blending
* Specialized LightGBM components
* TabICL as a small diversity component
* Rank-based ensemble blending
* Cross-validation-based prediction generation

### What did not provide meaningful improvement

* Some feature-engineering approaches
* Some segment interaction experiments
* Simply adding another model without sufficient diversity
* Treating foundation models as automatic replacements for GBDTs

---

# 👨‍💻 Author

**Garv Goel**

GitHub: **[@Garv3068](https://github.com/Garv3068)**

---

# ⭐ Project Philosophy

> **Don't chase more models. Chase better information.**

The objective of this project is not simply to train the largest number of models possible.

It is to systematically discover complementary sources of predictive information and combine them into a robust, high-performing ensemble.

---

## 📜 Project Status

**Active Kaggle Competition Project**

Current stage:

```text
Phase 5 — Advanced Ensemble
             ↓
       AutoGluon Next
             ↓
        Final Freeze
```

🚀 **Still optimizing.**
