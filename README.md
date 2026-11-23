# 🎬 A Two-Stage Funnel Recommendation Architecture

> **Integrating Candidate Generation with Deep Learning Scoring on the MovieLens Dataset**

**Course:** Recommendation Systems — MSc Computer Engineering, Atatürk University  
**Student:** Hüseyin Sefa Akçay (26040201005)  
**Instructor:** Asst. Prof. Resul Tugay  

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Dataset](#-dataset)
4. [Methods](#-methods)
5. [Evaluation](#-evaluation)
6. [Project Structure](#-project-structure)
7. [Installation & Setup](#-installation--setup)
8. [Usage](#-usage)
9. [Results](#-results)
10. [Timeline & Deliverables](#-timeline--deliverables)
11. [References](#-references)

---

## 🎯 Project Overview

Modern industrial recommendation systems face a fundamental trade-off between **computational latency** and **ranking expressiveness** when dealing with large item catalogs. Platforms such as YouTube, Netflix, and Amazon resolve this using a **Funnel Architecture** — a multi-stage pipeline that progressively narrows the item space from millions of candidates to a small, personalized ranked list.

This project implements a small-scale academic simulation of such a two-stage funnel on the **MovieLens 1M** dataset and rigorously evaluates whether this architecture preserves the accuracy of single-stage deep learning models.

### Problem Statement

> *Given a user's historical rating interactions, generate a personalized Top-K list of movies that maximizes recommendation quality (Precision, Recall, NDCG), while simulating the retrieval-then-rank paradigm used in production recommender systems.*

### Goals

| Goal | Description |
|------|-------------|
| **Stage 1** | Narrow the full 3,706-movie catalog to a Top-N candidate pool per user using fast retrieval models (SVD + CBF) |
| **Stage 2** | Re-rank the candidate pool using a deep Neural Collaborative Filtering (NeuMF) model |
| **Comparison** | Rigorously compare 3 configurations: SVD-only, NCF-only, and the full two-stage funnel |
| **Analysis** | Study the effect of candidate pool size, data sparsity, and cold-start behavior on funnel performance |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERACTION HISTORY                     │
│               (MovieLens 1M — Temporal Train Split)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────▼───────────────┐
           │     STAGE 1: CANDIDATE        │
           │        GENERATION             │
           │  ┌─────────────────────────┐  │
           │  │  SVD (Matrix Factorize) │  │
           │  │  Recall@N per user      │  │
           │  └────────────┬────────────┘  │
           │               │  UNION        │
           │  ┌────────────▼────────────┐  │
           │  │   CBF (Genre Profiles)  │  │
           │  │   Cosine Similarity     │  │
           │  └────────────┬────────────┘  │
           │               │               │
           │      Top-N Candidate Pool     │
           │      (N = 50 / 100 / 200)     │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │    STAGE 2: SCORING &         │
           │        RANKING                │
           │  ┌─────────────────────────┐  │
           │  │  NeuMF (Neural MF)      │  │
           │  │  GMF + MLP branches     │  │
           │  │  BCE Loss, Adam, neg.   │  │
           │  │  sampling               │  │
           │  └────────────┬────────────┘  │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │      FINAL TOP-K LIST         │
           │   (K = 5 / 10 / 20)           │
           └───────────────────────────────┘
```

### NeuMF Model Architecture

```
User ID ──► User Embedding (MF dim=64) ──┐
Item ID ──► Item Embedding (MF dim=64) ──┤──► Element-wise Product ──► GMF Output ─┐
                                          │                                           ├──► FC → Sigmoid
User ID ──► User Embedding (MLP dim=64) ─┤──► Concat ──► [256→128→64→32] ─────────►┘
Item ID ──► Item Embedding (MLP dim=64) ─┘
```

---

## 📊 Dataset

**MovieLens 1M** — a classic benchmark for recommender systems research.

| Attribute | Value |
|-----------|-------|
| Source | [grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/) |
| Users | 6,040 |
| Movies | 3,706 |
| Ratings | 1,000,209 |
| Rating Scale | 1–5 stars |
| Sparsity | ~95.5% |
| Time Range | 2000–2003 |

**Files:**
- `ratings.dat` — `UserID::MovieID::Rating::Timestamp`
- `movies.dat` — `MovieID::Title::Genres`
- `users.dat` — `UserID::Gender::Age::Occupation::Zip`

**Train / Validation / Test Split:**  
Temporal split per user: last 10% of ratings → test, previous 10% → validation, remaining 80% → train. This prevents data leakage and simulates a realistic evaluation scenario.

---

## 🔬 Methods

### Stage 1A — SVD (Matrix Factorization)

- **Library:** `scikit-surprise` (SVD implementation)
- **Latent Factors:** 100
- **Learning:** Stochastic Gradient Descent
- **Role:** Generate Top-N candidates ranked by predicted rating score
- **Key Metric:** `Recall@N` — what fraction of test items appear in the candidate pool?

### Stage 1B — Content-Based Filtering (CBF)

- **Features:** Genre one-hot encoding (18 genres in MovieLens 1M)
- **Similarity:** Cosine similarity between user genre profile and item genre vectors
- **Role:** Supplement SVD candidates, especially for cold-start items
- **Combination:** Union of SVD and CBF candidates

### Stage 2 — NeuMF (Neural Collaborative Filtering)

Based on: *He et al., "Neural Collaborative Filtering," WWW 2017*

- **Architecture:** Fusion of GMF (Generalized Matrix Factorization) and MLP branches
- **Training:** Binary Cross-Entropy loss with negative sampling (4 negatives per positive)
- **Optimizer:** Adam (lr=0.001)
- **Framework:** PyTorch
- **Input:** User-item pairs from the candidate pool (not the full catalog)
- **Output:** Click probability scores for ranking

### Comparison Matrix

| Configuration | Stage 1 | Stage 2 | Role |
|---------------|---------|---------|------|
| **Baseline A** | SVD (full catalog) | — | Traditional MF baseline |
| **Baseline B** | Full catalog | NCF | Deep learning upper bound |
| **Proposed** | SVD + CBF (Top-N) | NCF | Two-stage funnel |

---

## 📏 Evaluation

All models are evaluated on the same temporal test split with the following metrics:

| Metric | Symbol | Description |
|--------|--------|-------------|
| Precision at K | P@K | Fraction of recommended items that are relevant |
| Recall at K | R@K | Fraction of relevant items that are recommended |
| Normalized DCG | NDCG@K | Ranking quality weighted by position |
| Mean Avg. Precision | mAP | Average precision across all relevant items |
| Area Under ROC | AUC | Binary classification performance |
| Root Mean Squared Error | RMSE | Rating prediction error (SVD baseline) |
| Mean Absolute Error | MAE | Average absolute rating error (SVD baseline) |

**K values:** {5, 10, 20}  
**Candidate pool sizes (N):** {50, 100, 200}

---

## 📁 Project Structure

```
MovieRecSysProject/
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                     ← MovieLens 1M raw files (not tracked by git)
│   │   ├── movies.dat
│   │   ├── ratings.dat
│   │   └── users.dat
│   └── processed/               ← Preprocessed splits (auto-generated)
│
├── notebooks/
│   ├── 01_EDA.ipynb             ← Exploratory Data Analysis
│   ├── 02_stage1_svd.ipynb      ← SVD Candidate Generation
│   ├── 03_stage1_cbf.ipynb      ← Content-Based Filtering
│   ├── 04_stage2_ncf.ipynb      ← Neural Collaborative Filtering
│   ├── 05_funnel_pipeline.ipynb ← End-to-end Two-Stage Funnel
│   └── 06_evaluation.ipynb      ← Comparative Evaluation
│
├── src/
│   ├── data/
│   │   ├── loader.py            ← Data loading + temporal split
│   │   └── preprocessor.py     ← Feature engineering + encoding
│   ├── models/
│   │   ├── svd_model.py         ← SVD candidate generator
│   │   ├── cbf_model.py         ← Content-based filtering
│   │   ├── ncf_model.py         ← NeuMF PyTorch model + trainer
│   │   └── funnel_pipeline.py  ← Two-stage orchestrator
│   ├── evaluation/
│   │   └── metrics.py           ← All evaluation metrics (from scratch)
│   └── utils/
│       └── helpers.py           ← Common utilities, seed setting, logging
│
├── experiments/
│   ├── config.yaml              ← All hyperparameters (single source of truth)
│   └── results/                 ← JSON/CSV experiment results
│
├── reports/
│   ├── intermediate_report/     ← IEEE 5-6 page report (02 Dec deadline)
│   └── final_report/            ← IEEE final report (02 Jan deadline)
│
└── demo/
    └── app.py                   ← Streamlit interactive demo (optional)
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/MovieRecSysProject.git
cd MovieRecSysProject
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download MovieLens 1M Dataset

```bash
# Option A — Manual
# Download from: https://grouplens.org/datasets/movielens/1m/
# Unzip ml-1m.zip and place the ml-1m/ folder under data/raw/

# Option B — Automated
python scripts/download_data.py
```

Expected structure after setup:
```
data/raw/ml-1m/
├── movies.dat
├── ratings.dat
└── users.dat
```

---

## 🚀 Usage

### Run All Notebooks (Recommended)

Open JupyterLab and run notebooks in order:

```bash
jupyter lab
```

| Step | Notebook | Output |
|------|----------|--------|
| 1 | `01_EDA.ipynb` | Dataset statistics, distribution plots |
| 2 | `02_stage1_svd.ipynb` | SVD model, Recall@N curves |
| 3 | `03_stage1_cbf.ipynb` | Genre similarity matrix, CBF candidates |
| 4 | `04_stage2_ncf.ipynb` | NeuMF training curves, ranking metrics |
| 5 | `05_funnel_pipeline.ipynb` | End-to-end funnel evaluation |
| 6 | `06_evaluation.ipynb` | Side-by-side comparison tables & plots |

### Run as Python Scripts

```bash
# Data preprocessing
python -m src.data.loader

# Train SVD
python -m src.models.svd_model --config experiments/config.yaml

# Train NCF
python -m src.models.ncf_model --config experiments/config.yaml

# Full funnel evaluation
python -m src.models.funnel_pipeline --config experiments/config.yaml
```

---

## 📈 Results

> Results will be updated as experiments are completed.

### Stage 1 — Candidate Coverage (Recall@N)

| Method | Recall@50 | Recall@100 | Recall@200 |
|--------|-----------|------------|------------|
| SVD | — | — | — |
| CBF | — | — | — |
| SVD + CBF (Union) | — | — | — |

### Stage 2 — Ranking Quality

| Model | P@10 | R@10 | NDCG@10 | mAP | AUC |
|-------|------|------|---------|-----|-----|
| SVD-only (Baseline A) | — | — | — | — | — |
| NCF-only (Baseline B) | — | — | — | — | — |
| **Two-Stage Funnel** | — | — | — | — | — |

---

## 📅 Timeline & Deliverables

| Date | Deliverable | Status |
|------|-------------|--------|
| 29 Oct 2026 | Project Topic | ✅ Accepted |
| 04 Nov 2026 | Project Proposal (IEEE 1-2 pages) | ✅ Submitted |
| **02 Dec 2026** | **Intermediate Report (IEEE 5-6 pages)** | ⏳ In Progress |
| 23 Dec 2026 | Demo/Presentation + All Code Submission | 🔜 Upcoming |
| 23–30 Dec 2026 | Live Demo/Presentation | 🔜 Upcoming |
| 02 Jan 2027 | Final Report | 🔜 Upcoming |

---

## 📚 References

1. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). **Neural collaborative filtering.** *Proceedings of the 26th International Conference on World Wide Web (WWW 2017)*, 173–182.

2. Koren, Y., Bell, R., & Volinsky, C. (2009). **Matrix factorization techniques for recommender systems.** *Computer*, 42(8), 30–37.

3. Harper, F. M., & Konstan, J. A. (2015). **The MovieLens datasets: History and context.** *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), 1–19.

4. Yi, X., Yang, J., Hong, L., Cheng, D. Z., Heldt, L., Kumthekar, A., ... & Chi, E. (2019). **Sampling-bias-corrected neural modeling for large corpus item recommendations.** *Proceedings of RecSys 2019*.

5. Covington, P., Adams, J., & Sargin, E. (2016). **Deep neural networks for YouTube recommendations.** *Proceedings of RecSys 2016*, 191–198.

6. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). **BPR: Bayesian personalized ranking from implicit feedback.** *Proceedings of UAI 2009*.

---

## 🔧 Configuration

All hyperparameters are centralized in [`experiments/config.yaml`](experiments/config.yaml):

```yaml
stage1:
  svd:
    n_factors: 100
    n_epochs: 20
  top_n_candidates: 100

stage2:
  ncf:
    mf_dim: 64
    mlp_layers: [256, 128, 64, 32]
    lr: 0.001
    n_negatives: 4
    n_epochs: 20

evaluation:
  K_values: [5, 10, 20]
```

---

## 📝 Version Control

This project uses Git for version control as required by the course.

```
main      ← Stable, submission-ready
└── dev   ← Active development
    ├── feature/data-pipeline
    ├── feature/svd-stage1
    ├── feature/ncf-stage2
    └── feature/funnel-integration
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) format:  
`feat: add NeuMF model with GMF and MLP branches`

---

<p align="center">
  <i>Recommendation Systems Course — MSc Computer Engineering — Atatürk University — 2026</i>
</p>
