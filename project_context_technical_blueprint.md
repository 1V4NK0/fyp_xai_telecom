# Explainable AI (xAI) for Automated Telecom Networks
**Department of Computer Science & Applied Physics — Atlantic Technological University (ATU)**  
**Module:** Applied Project & Minor Dissertation (Level 8, 15 ECTS Credits)  
**Student:** Ivan Cherkes  
**Academic Supervisor:** Dr. John Healy  
**Industry Partner / Context:** Ericsson  

---

## 1. Project Overview & Problem Statement

### 1.1 Context
Next-generation cellular networks (5G and evolving 6G architectures) increasingly depend on artificial intelligence and machine learning to manage radio resources, perform dynamic network slicing, and execute closed-loop autonomous decisions (such as cell handovers and traffic steering) without human latency.

However, standard high-capacity machine learning algorithms function as opaque "black boxes." When an automated network controller denies a service level agreement (SLA), degrades an active session, or steers traffic between base stations, radio network engineers require verifiable rationale. Telecommunications operators and regulatory frameworks mandate transparency, auditability, and deterministic fault isolation before autonomous decision controllers can be fully trusted in production deployments.

### 1.2 Core Goal
This project investigates and benchmarks post-hoc, model-agnostic **Explainable Artificial Intelligence (xAI)** techniques—specifically **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-Agnostic Explanations)**—applied to telecommunications telemetry data. The primary objective is to evaluate how effectively these frameworks explain automated 5G network decisions (e.g., network slice classification and traffic routing), comparing their computational overhead, explanation stability, and operational interpretability.

---

## 2. Key Terminology & Domain Reference

### 2.1 Telecom Telemetry Metrics (Model Inputs / Features)
* **Latency ($ms$):** Round-trip packet transmission time from User Equipment (UE) to the core network endpoint. Critical for delay-sensitive applications.
* **Jitter ($ms$):** Statistical variance in packet arrival latency over time.
* **Packet Loss Rate (%):** Percentage of transmitted packets that fail to reach their destination due to congestion, signal degradation, or buffer exhaustion.
* **Throughput / Data Rate ($Mbps$):** Delivered volume of data per second over an active channel.
* **SINR (Signal-to-Interference-plus-Noise Ratio, $dB$):** Ratio of desired radio signal power to combined interference from neighboring cells plus thermal noise.
* **RSRP (Reference Signal Received Power, $dBm$):** Measured physical power level received by a mobile terminal from a specific antenna.
* **GBR (Guaranteed Bit Rate):** Quality of Service (QoS) classification ensuring minimum dedicated bandwidth for time-critical services, contrasted with Non-GBR (best-effort traffic).

### 2.2 Autonomous Network Operations (Model Outputs / Targets)
* **5G Network Slicing:** Virtualized partition of physical cellular infrastructure into distinct logical networks with bespoke QoS profiles:
  * **eMBB (Enhanced Mobile Broadband):** High-data-rate applications (e.g., 4K/8K video streaming, AR/VR, bulk downloads).
  * **URLLC (Ultra-Reliable Low-Latency Communication):** Mission-critical services demanding ultra-low latency ($\le 10\text{ ms}$) and extreme reliability (e.g., autonomous driving, remote robotic surgery, industrial control).
  * **mMTC (Massive Machine-Type Communications):** Scaled connectivity for high-density, low-power Internet of Things (IoT) sensors and smart metering.
* **Handover / Handoff:** Seamless migration of an active data call/stream from one gNodeB (5G base station) or sector to another.
* **Traffic Steering:** Dynamically rerouting IP packet flows across alternate cell sectors or spectrum bands to mitigate localized congestion.
* **SLA (Service Level Agreement):** Strict contractual performance targets (maximum allowable latency, minimum uptime, packet error budget).

---

## 3. Explainable AI Foundations: SHAP vs. LIME

### 3.1 LIME (Local Interpretable Model-Agnostic Explanations)
* **Foundational Concept:** Evaluates decisions locally. While global decision surfaces of non-linear models are complex, small localized neighborhoods around a single inference behave approximately linear.
* **Mechanism:**
  1. Takes an input instance $x$ and generates synthetic perturbations ($x'$) around it.
  2. Passes perturbed samples through the black-box model to gather output probabilities.
  3. Weights perturbed instances inversely proportional to their Euclidean distance from $x$.
  4. Trains an interpretable surrogate model (e.g., Ridge Regression) over the weighted perturbations.
* **Visual Output:** Horizontal feature contribution bar charts displaying positive and negative feature attribution weights for that discrete inference.
* **Limitations:** Sampling variability means multiple runs on the exact same record can produce slightly divergent attribution weights; sensitive to perturbation kernel width.

### 3.2 SHAP (SHapley Additive exPlanations)
* **Foundational Concept:** Grounded in Cooperative Game Theory (Lloyd Shapley, 1953). Treats each telemetry feature as a "player" in a coalition and assigns payout credit based on marginal contribution across all subsets.
* **Mechanism:**
  * Computes the marginal difference in prediction when feature $i$ is present versus absent across all possible feature subsets $S \subseteq F \setminus \{i\}$:
    $$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
  * Yields additive attribution:
    $$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$
    where $\phi_0$ is the base expected model prediction $E[f(x)]$.
* **Visual Outputs:**
  * **Waterfall / Force Plot:** Local explanation tracing step-by-step feature pushes from baseline to final output probability.
  * **Summary / Beeswarm Plot:** Global visualization plotting thousands of instances, revealing feature impact direction and distribution across the entire network dataset.
* **Key Implementation:** Utilizes the optimized `TreeExplainer` for decision trees and ensembles, reducing exponential complexity to polynomial time.

### 3.3 Core Comparison Dimensions for Research
* **Fidelity & Agreement:** Degree of consensus between SHAP top features and LIME top features for anomalous network events.
* **Execution Latency:** Run-time computation time ($ms$ per sample) to assess whether explainability pipelines can operate near real-time in telecommunications control planes.
* **Stability:** Run-to-run explanation variance under identical input parameters.

---

## 4. Technical Architecture & Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                            │
│  - DeepSlice Telemetry Benchmark CSV (63,167 records)        │
│  - Pandas & NumPy (Preprocessing, Encoding, Feature Slicing)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      ML Engine Layer                        │
│  - Scikit-Learn Supervised Classification Pipelines         │
│  - Random Forest Classifier (Ensemble Baseline)             │
│  - Train/Test Splitting, Metrics (F1, Precision, Recall)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Explainability Layer (xAI)                  │
│  - SHAP: shap.TreeExplainer (Local Waterfall, Beeswarm)     │
│  - LIME: lime.lime_tabular.LimeTabularExplainer             │
│  - Performance Benchmarking (Latency, Stability Metrics)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Presentation & UI                        │
│  - Matplotlib / Seaborn (Dissertation-ready vector figures) │
│  - Streamlit Dashboard (Interactive slice inspector demo)   │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Component | Version / Tool | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11+ | Primary programming language |
| **Data Processing** | Pandas, NumPy | Latest stable | Data loading, manipulation, and encoding |
| **Model Development** | Scikit-Learn | Latest stable | Tabular classification and model validation |
| **Explainability** | `shap`, `lime` | Official PyPI packages | Model-agnostic local/global explanation generation |
| **Visualization** | Matplotlib, Seaborn | Latest stable | Statistical plotting and thesis figures |
| **Interface Demo** | Streamlit | Latest stable | Interactive dashboard for Viva demonstration |
| **Version Control** | Git / GitHub | Remote: `fyp_xai_telecom` | Continuous development tracking and audit log |
| **Academic Writing** | Overleaf (LaTeX) | Department Template | Minor Dissertation document (10k–15k words) |

---

## 5. Primary Dataset Specifications

* **Source:** DeepSlice Benchmark Dataset (`deepslice_data.csv`)
* **Academic Reference:** *"DeepSlice: A Deep Learning Approach to Network Slicing in 5G Cellular Networks"* (Thantharate et al.)
* **Total Records:** 63,167 instances
* **Missing Values:** 0 null/NaN values across all fields
* **Input Features ($X$):**
  * `Packet delay` (Continuous numeric: 10ms to 300ms)
  * `Packet Loss Rate` (Continuous float: $10^{-6}$ to $10^{-2}$)
  * `GBR` (Categorical: Guaranteed Bit Rate status)
  * `Use Case` (Categorical: Smartphone, Healthcare, Industry 4.0, AR/VR, Smart Transportation, Public Safety, IoT)
  * `Technology Supported` (Categorical: LTE, 5G, IoT)
  * `LTE/5g Category` (Device capability classification)
  * Contextual variables: `Day`, `Time`
* **Target Label ($y$):**
  * `slice Type` (3 classes: `eMBB` [33,599], `URLLC` [14,784], `mMTC` [14,784])

---

## 6. Academic Deliverables & Assessment Criteria

* **Module Weight:** 15 ECTS Credits (25% of final Year 4 degree award classification).
* **Workload Expectation:** ~12–15 hours/week of total combined student effort (3x standard 5-credit module weighting).

```
Semester 1 Review (10%)          April 2027 Submission (45%)         May 2027 Examination (45%)
─────────────────────────        ───────────────────────────         ──────────────────────────
• Christmas Seminar              • Minor Dissertation LaTeX PDF       • A0 Poster Presentation
• 5 slides in 5 minutes            (10,000–15,000 words, ~35–50 p.)  • 10-minute Viva Voce Talk
• Minimal skeletal prototype     • GitHub Codebase & Commits         • 5–10 minutes Panel Q&A
• Q&A with programme board       • Recorded Screencast Demo          • Independent grading panel
```

### Key Dissertation Chapters (Overleaf / LaTeX)
1. **Introduction:** Problem context, autonomous network motivation, project objectives, research questions.
2. **Technology Review:** Mathematical and conceptual foundations of Black-Box ML, Shapley Values, LIME local approximations, and review of related 5G xAI literature.
3. **System Design & Architecture:** End-to-end data pipelines, ML classification engine architecture, explainability layer design, and interactive UI interface specification.
4. **Implementation:** Detailed technical description of data preprocessing, model training protocols, and explainer integration.
5. **System Evaluation & Critical Reflection:** Model accuracy/F1 performance, quantitative SHAP vs. LIME comparison (latency, stability, feature agreement), operational impact on telecom engineers, and project limitations.
6. **Conclusions & Future Work:** Summary of findings, standardization implications, and extension to online streaming pipelines.

---

## 7. Semester 1 Execution Schedule

* **Week 1 (Mid-October):** 
  * Finalize proposal text and submit to supervisor / Moodle.
  * Initialize GitHub repository (`fyp_xai_telecom`) and Overleaf LaTeX workspace.
* **Week 2 (Late October):** 
  * Python data fundamentals and environment verification (`venv` / dependencies).
  * Load and inspect `deepslice_data.csv` using Pandas.
* **Week 3 (Early November):** 
  * Exploratory Data Analysis (EDA) script: categorical encoding, summary statistics, class distribution plots.
  * Establish baseline data preprocessing pipeline.
* **Week 4 (Mid-November):** 
  * Implement Scikit-Learn `RandomForestClassifier` with train/test split.
  * Evaluate baseline classification accuracy, confusion matrix, and precision/recall.
* **Week 5 (Late November):** 
  * Integrate `shap.TreeExplainer` on the trained model.
  * Generate and export initial waterfall plot and global beeswarm plot.
* **Week 6 (Late Nov – Early Dec):** 
  * Integrate `lime.lime_tabular.LimeTabularExplainer` on identical test samples.
  * Create side-by-side local explanation visualization.
* **Week 7 (Early December):** 
  * Assemble skeletal prototype script connecting dataset $\rightarrow$ model $\rightarrow$ xAI outputs.
  * Draft Introduction chapter outline in Overleaf.
* **Week 8 (Mid-December):** 
  * Prepare 5-slide Christmas presentation deck.
  * Deliver Christmas Seminar presentation (10% milestone).