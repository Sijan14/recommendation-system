"""
svd_model.py
SVD matrix factorization using scipy truncated SVD.
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


def run_svd(matrix_demeaned, user_means, n_factors):
    """
    Decompose the mean-centered matrix and reconstruct predicted ratings.
    Returns a DataFrame of predicted ratings (users x movies).
    """
    max_factors = min(matrix_demeaned.shape) - 1
    if n_factors >= max_factors:
        raise ValueError(f'n_factors ({n_factors}) must be < {max_factors}')

    U, sigma, Vt = svds(matrix_demeaned.values.astype(float), k=n_factors)
    predicted    = np.dot(np.dot(U, np.diag(sigma)), Vt)
    predicted   += user_means.values.reshape(-1, 1)

    return pd.DataFrame(
        predicted,
        index=matrix_demeaned.index,
        columns=matrix_demeaned.columns
    )


def tune_svd(matrix_demeaned, user_means, val_df, global_mean,
             n_factors_list=(5, 10, 20, 50, 100)):
    """
    Tune number of latent factors on a validation sample.
    Returns the best n_factors value.
    """
    from sklearn.metrics import mean_squared_error
    max_factors = min(matrix_demeaned.shape) - 1
    best_n, best_rmse = n_factors_list[0], float('inf')

    for n in n_factors_list:
        if n >= max_factors:
            print(f'  n_factors={n} exceeds limit ({max_factors}), skipping.')
            continue

        pred_matrix = run_svd(matrix_demeaned, user_means, n)
        preds, actuals = [], []

        for _, row in val_df.iterrows():
            uid, mid = row['userId'], row['movieId']
            if uid in pred_matrix.index and mid in pred_matrix.columns:
                preds.append(pred_matrix.loc[uid, mid])
            else:
                preds.append(global_mean)
            actuals.append(row['rating'])

        rmse = np.sqrt(mean_squared_error(actuals, preds))
        print(f'  n_factors={n:>4}  RMSE: {rmse:.4f}')
        if rmse < best_rmse:
            best_rmse, best_n = rmse, n

    print(f'Best n_factors: {best_n}')
    return best_n


def predict_svd(pred_matrix, user_id, movie_id, global_mean):
    """Return SVD predicted rating for a single (user, movie) pair."""
    if user_id in pred_matrix.index and movie_id in pred_matrix.columns:
        return pred_matrix.loc[user_id, movie_id]
    return global_mean
