# Explainable AI (xAI) for Automated Telecom Networks (`fyp_xai_telecom`)

> **Level 8 Final Year Project & Minor Dissertation**  
> Department of Computer Science & Applied Physics — Atlantic Technological University (ATU)  
> **Student:** Ivan Cherkes | **Academic Supervisor:** Dr. John Healy | **Industry Context:** Ericsson

---

## Overview

Modern 5G and emerging 6G networks rely on machine learning for automated radio resource management, traffic steering, and dynamic **network slicing** (`eMBB`, `URLLC`, `mMTC`). Because standard ML classifiers act as opaque black boxes, telecommunications operators require transparent, auditable, and fast explanations before trusting automated decisions in production SLA environments.

This project investigates and benchmarks post-hoc **Explainable Artificial Intelligence (xAI)** frameworks—primarily **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-Agnostic Explanations)—on 5G network slicing telemetry.

### Key Research Objectives
1. **Model Diversity & Rigour:** Evaluate and compare multiple ML classifier families (**Random Forest**, **AdaBoost**, **SVM**, **MLP**, **Naive Bayes**, **K-Nearest Neighbors**) using **Stratified N-Fold Cross-Validation**.
2. **Robustness & Generalisation:** Investigate baseline overfitting on clean simulated telemetry via **controlled noise stress-testing** and deeper Exploratory Data Analysis (EDA).
3. **xAI Benchmarking (SHAP vs. LIME):** Compare explanation fidelity, feature attribution agreement, execution latency (ms/sample), and perturbation stability across model types.
4. **Interactive Inspection:** Deliver a prototype dashboard (Streamlit) allowing telecom engineers to inspect per-slice predictions alongside live local explanations.

---

## Dataset & Provenance

* **File:** `data/deepslice_data.csv`
* **Scale:** 63,167 records | 0 missing values | 8 input features (`Use Case`, `LTE/5g Category`, `Technology Supported`, `Day`, `Time`, `GBR`, `Packet Loss Rate`, `Packet delay`)
* **Target (`slice Type`):**
  * `eMBB` (Enhanced Mobile Broadband): 33,599 rows (53.19%)
  * `URLLC` (Ultra-Reliable Low-Latency Communication): 14,784 rows (23.40%)
  * `mMTC` (Massive Machine-Type Communications): 14,784 rows (23.40%)
* **Download Source (Kaggle Mirror):** https://www.kaggle.com/datasets/puspakmeher/networkslicing/data
* **Academic Origin:** DeepSlice benchmark dataset — *"DeepSlice: A Deep Learning Approach to Network Slicing in 5G Cellular Networks"* (Thantharate et al.)
* **Note on Trustworthiness & Limitations:** DeepSlice is a **simulated** 5G telemetry benchmark built around 3GPP QoS profiles rather than live Ericsson production traffic. Its deterministic synthetic rules explain why an unperturbed baseline Random Forest achieves 100% test accuracy—motivating the cross-validation, expanded EDA, and noise-injection robustness experiments in this repository.

---

## Repository Structure

```text
fyp_xai_telecom/
├── 01_data_exploration.py                # Initial EDA & per-slice summary statistics
├── 02_train_baseline_model.py            # Stratified 80/20 split, One-Hot Encoding & RF baseline
├── 03_shap_explainability.py             # TreeSHAP local waterfall visualisations
├── data/
│   ├── deepslice_data.csv                # DeepSlice 5G network slicing dataset
│   └── slice_type_summary.csv            # Generated summary statistics table
├── models/
│   ├── baseline_random_forest_model.joblib
│   └── model_features.joblib
├── outputs/
│   └── figures/                          # Saved SHAP waterfall visualisations (.png)
├── scrapbook/
│   └── 2026-11-10_ericsson_meeting_notes.txt  # Ideas, meeting logs, and experiment backlog
├── PROJECT_PROGRESS_LOG.md               # Milestone & assessment tracking
├── project_context_technical_blueprint.md
└── README.md
```

---

## Technology Stack

* **Language:** Python 3.11+
* **Data & ML:** `pandas`, `scikit-learn`, `joblib`
* **Explainability:** `shap`, `lime`
* **Visualisation:** `matplotlib`, `seaborn`

---

## Getting Started

1. **Clone the repository and activate your virtual environment:**
   ```bash
   git clone https://github.com/1V4NK0/fyp_xai_telecom.git
   cd fyp_xai_telecom
   source venv/bin/activate
   ```

2. **Run the pipeline scripts sequentially:**
   ```bash
   python 01_data_exploration.py
   python 02_train_baseline_model.py
   python 03_shap_explainability.py
   ```

---

## Progress & Roadmap

- [x] **Phase 1:** Initial Data Exploration (`01_data_exploration.py`) & summary export
- [x] **Phase 2:** Baseline Random Forest classifier (`02_train_baseline_model.py`)
- [x] **Phase 3 (Initial):** Local SHAP Waterfall explainability (`03_shap_explainability.py`)
- [x] **Milestone:** Kickoff / Progress Review with Dr. John Healy & Ericsson Engineers (Juan Jose Agullo, Kalyan Desaraju)
- [ ] **Phase 3b:** Expanded EDA (feature distributions, class separability, correlations)
- [ ] **Phase 4:** Multi-model comparison (Random Forest, AdaBoost, SVM, MLP, Naive Bayes, KNN) + Stratified N-Fold Cross-Validation
- [ ] **Phase 5:** Overfitting check & controlled noise stress-testing
- [ ] **Phase 6:** Global SHAP Beeswarm + LIME explainability & SHAP-vs-LIME quantitative benchmarking
- [ ] **Phase 7:** Christmas Seminar Prototype (Semester 1 Review)
- [ ] **Phase 8:** Interactive Streamlit Dashboard & Minor Dissertation (LaTeX)
