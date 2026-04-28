# Movie Recommendation System

**Course:** CSIT 557 — Advanced Techniques in Data Science  
**Semester:** Spring 2026  
**Author:** Md Allama Ikbal Sijan  

---

## Project Overview

This project builds a movie recommendation system from scratch using the CiaoDVD dataset. It implements and compares four recommendation algorithms: a global average baseline, user-based collaborative filtering, item-based collaborative filtering, and SVD matrix factorization. The project follows a vibe coding approach, using AI-assisted development throughout.

---

## Dataset

**CiaoDVD** — crawled from dvd.ciao.co.uk in December 2013.

| File | Description |
|------|-------------|
| `ratings.txt` | userId, movieId, rating (1–5) |
| `trust.txt` | trustorId, trusteeId, trustRating |

Download: https://guoguibing.github.io/librec/datasets.html  
Place downloaded files in `data/raw/`.

**Key statistics after cleaning:**

| Metric | Value |
|--------|-------|
| Total users | 1,508 |
| Total movies | 2,071 |
| Total ratings | 35,497 |
| Matrix sparsity | 98.86% |
| Median ratings per user | 15 |
| Median ratings per movie | 2 |

---

## Repository Structure

```
recommendation-system/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # Original dataset files (not committed)
│   └── processed/    # Cleaned data outputs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cf_model.ipynb
│   ├── 03_svd_model.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── cf_model.py
│   ├── svd_model.py
│   └── evaluation.py
├── results/
│   ├── figures/
│   ├── metrics_cf.csv
│   ├── metrics_all.csv
│   └── metrics_topk.csv
└── report/
    └── final_report.pdf
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Sijan14/recommendation-system.git
cd recommendation-system

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

Run the notebooks in order:

```bash
# Step 1: Data loading and EDA
jupyter notebook notebooks/01_eda.ipynb

# Step 2: Collaborative filtering models
jupyter notebook notebooks/02_cf_model.ipynb

# Step 3: SVD matrix factorization
jupyter notebook notebooks/03_svd_model.ipynb

# Step 4: Evaluation and comparison
jupyter notebook notebooks/04_evaluation.ipynb
```

---

## Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| Global Average | Baseline | Predicts the global mean rating for all pairs |
| User-Based CF | Collaborative Filtering | Recommends based on similar users (cosine similarity) |
| Item-Based CF | Collaborative Filtering | Recommends based on similar items (cosine similarity) |
| SVD | Matrix Factorization | Learns latent user and item factors via truncated SVD |

---

## Results

### RMSE and MAE

| Model | RMSE | MAE |
|-------|------|-----|
| Global Average | 0.8122 | 0.6603 |
| User-Based CF | 0.8260 | 0.6605 |
| Item-Based CF | 0.7292 | 0.5726 |
| SVD (k=10) | **0.7235** | **0.5654** |

### Precision@10 and Recall@10 (relevance threshold: rating >= 4.0)

| Model | Precision@10 | Recall@10 |
|-------|-------------|----------|
| Global Average | 0.2366 | 0.6362 |
| Item-Based CF | **0.2502** | **0.6957** |
| SVD (k=10) | 0.2488 | 0.6863 |

SVD achieves the best rating prediction accuracy (RMSE/MAE). Item-Based CF achieves the best ranking quality (Precision@10 and Recall@10).

---

## References

- Guo, G., Zhang, J., & Yorke-Smith, N. (2013). A novel Bayesian similarity measure for recommender systems. *IJCAI*.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *IEEE Computer, 42*(8), 30–37.
- Hug, N. (2020). Surprise: A Python library for recommender systems. *JOSS, 5*(52), 2174.
