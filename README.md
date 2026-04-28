# Movie Recommendation System

**Course:** CSIT 557 — Advanced Techniques in Data Science  
**Semester:** Spring 2026  
**Author:** Md Allama Ikbal Sijan

---

## Project Overview

This project builds a movie recommendation system from scratch using the CiaoDVD dataset. It implements and compares multiple recommendation algorithms including collaborative filtering and SVD matrix factorization.

---

## Dataset

**CiaoDVD** — crawled from dvd.ciao.co.uk in December 2013.

| File | Description | Size |
|------|-------------|------|
| `movie-ratings.txt` | userId, movieId, categoryId, reviewId, rating (1–5), date | 72,665 ratings |
| `trusts.txt` | trustorId, trusteeId, trustRating | 40,133 trust links |

Download: https://guoguibing.github.io/librec/datasets.html

Place downloaded files in `data/raw/`.

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
│   └── metrics.csv
└── report/
    └── final_report.pdf
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/recommendation-system.git
cd recommendation-system

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
# Step 1: EDA
jupyter notebook notebooks/01_eda.ipynb

# Step 2: Collaborative Filtering models
jupyter notebook notebooks/02_cf_model.ipynb

# Step 3: SVD model
jupyter notebook notebooks/03_svd_model.ipynb

# Step 4: Evaluation and comparison
jupyter notebook notebooks/04_evaluation.ipynb
```

---

## Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| Global Average | Baseline | Predicts the mean rating for all pairs |
| User-Based CF | Collaborative Filtering | Recommends based on similar users |
| Item-Based CF | Collaborative Filtering | Recommends based on similar items |
| SVD | Matrix Factorization | Learns latent user/item factors |

---

## Results

*To be updated after evaluation.*

| Model | RMSE | MAE |
|-------|------|-----|
| Global Average | — | — |
| User-Based CF | — | — |
| Item-Based CF | — | — |
| SVD | — | — |

---

## References

- Guo, G., Zhang, J., & Yorke-Smith, N. (2013). A novel Bayesian similarity measure for recommender systems. *IJCAI*.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *IEEE Computer, 42*(8), 30–37.
- Hug, N. (2020). Surprise: A Python library for recommender systems. *JOSS, 5*(52), 2174.
