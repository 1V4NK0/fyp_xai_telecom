# Project Progress Log & Persistent Technical State
**Project:** Explainable AI (xAI) for Automated Telecom Networks (Level 8 FYP)  
**Student & Driver:** Ivan Cherkes  
**Academic Supervisor:** Dr. John Healy  
**Industry Partner / Context:** Ericsson  

---

## 1. Project Objective & Core Architecture
* **Goal:** Benchmark model-agnostic Explainable AI techniques (**SHAP** vs. **LIME**) on 5G network telemetry to explain automated slicing decisions in terms of execution latency, explanation stability, and feature attribution fidelity.
* **Dataset:** DeepSlice Benchmark (`deepslice_data.csv`) — 63,167 records, 0 missing values, 8 input features, 1 target label (`slice Type`).
* **Environment:** Python 3.11+ virtual environment (`venv`) with `scikit-learn`, `shap`, `lime`, `pandas`, `joblib`, `matplotlib`, `seaborn`.

---

## 2. Completed Milestones

### Phase 01: Exploratory Data Analysis (EDA) [COMPLETED]
* **Script:** `01_data_exploration.py`
* **Artifact Generated:** `slice_type_summary.csv`
* **Key Findings:**
  * Target distribution: `eMBB` = 33,599 (53.19%), `mMTC` = 14,784 (23.40%), `URLLC` = 14,784 (23.40%).
  * Class imbalance detected (eMBB > 50%), requiring Macro F1 and Recall evaluation in addition to accuracy.
  * Telecom domain verification: URLLC delay is strictly 10 ms with near-zero loss ($10^{-6}$), confirming SLA physics.

### Phase 02: Baseline Machine Learning Pipeline [COMPLETED]
* **Script:** `02_train_baseline_model.py` (Authored by Ivan)
* **Pipeline Structure:**
  1. Loaded `deepslice_data.csv` with `index_col=0`.
  2. Separated features $X$ (8 columns) from target $y$ (`slice Type`).
  3. One-Hot Encoded text features using `pd.get_dummies(..., dtype=int)`, expanding $X$ from 8 to 23 purely numerical features.
  4. Performed stratified train/test split: 80% train (50,533 rows), 20% test (12,634 rows) using `random_state=42` and `stratify=y`.
  5. Trained `RandomForestClassifier(n_estimators=100, random_state=42)`.
  6. Evaluated baseline performance: **100.00% Test Accuracy**, Precision=1.00, Recall=1.00, F1=1.00 across all 3 classes.
  7. Persisted artifacts: `baseline_random_forest_model.joblib` and `model_features.joblib`.

---

## 3. Upcoming Milestones & Roadmap

* **Phase 03 (Week 3 - Current Next Step):** SHAP Explainability Engine
  * Script: `03_shap_explainability.py`
  * Load `baseline_random_forest_model.joblib`.
  * Initialize `shap.TreeExplainer`.
  * Compute local Shapley values and generate a **Waterfall Plot** for a critical sample (e.g., Healthcare / URLLC).
  * Compute global Shapley values and generate a **Beeswarm Summary Plot**.
* **Phase 04 (Week 4):** LIME Explainability Engine
  * Script: `04_lime_explainability.py`
  * Initialize `LimeTabularExplainer` on identical test samples.
  * Generate local feature attribution bar charts.
* **Phase 05 (Weeks 5-6):** Benchmarking & Comparative Study
  * Measure inference latency ($ms$/sample) for real-time 5G control planes.
  * Measure run-to-run explanation stability (perturbation variance).
  * Measure feature attribution agreement between SHAP and LIME.
* **Phase 06 (December):** Christmas Seminar Presentation (10% Milestone)
  * 5 slides in 5 minutes + minimal terminal demonstration.
* **Phase 07 (Semester 2):** Interactive Streamlit Web App (`app.py`).
* **Phase 08 (Semester 2):** Dissertation Writing (LaTeX / Overleaf) & Viva Voce Exam.
