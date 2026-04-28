"""
evaluation.py
RMSE, MAE, Precision@K, and Recall@K evaluation functions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(y_true, y_pred, model_name='Model'):
    """Print and return RMSE and MAE for a set of predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f'{model_name:<22} | RMSE: {rmse:.4f} | MAE: {mae:.4f}')
    return rmse, mae


def precision_recall_at_k(user_predictions, user_actuals, k, threshold=4.0):
    """
    Compute Precision@K and Recall@K for a single user.

    Parameters
    ----------
    user_predictions : list of floats — predicted ratings
    user_actuals     : list of floats — actual ratings
    k                : int — number of top items to consider
    threshold        : float — minimum rating to be considered relevant

    Returns
    -------
    precision, recall : float, float
    """
    sorted_idx     = np.argsort(user_predictions)[::-1][:k]
    top_k_actuals  = np.array(user_actuals)[sorted_idx]
    top_k_relevant = np.sum(top_k_actuals >= threshold)
    total_relevant = np.sum(np.array(user_actuals) >= threshold)

    precision = top_k_relevant / k
    recall    = top_k_relevant / total_relevant if total_relevant > 0 else 0.0
    return precision, recall


def evaluate_top_k(test_df, predict_fn, k=10, threshold=4.0, **predict_kwargs):
    """
    Compute mean Precision@K and Recall@K across all eligible users.

    Parameters
    ----------
    test_df     : DataFrame with columns userId, movieId, rating
    predict_fn  : callable(user_id, movie_id, **predict_kwargs) -> float
    k           : int
    threshold   : float — relevance threshold

    Returns
    -------
    mean_precision, mean_recall : float, float
    """
    precisions, recalls = [], []

    for user_id, group in test_df.groupby('userId'):
        if len(group) < k:
            continue

        actuals   = group['rating'].tolist()
        movie_ids = group['movieId'].tolist()
        preds     = [predict_fn(user_id, mid, **predict_kwargs) for mid in movie_ids]

        p, r = precision_recall_at_k(preds, actuals, k, threshold)
        precisions.append(p)
        recalls.append(r)

    mean_p = float(np.mean(precisions)) if precisions else 0.0
    mean_r = float(np.mean(recalls))    if recalls    else 0.0
    return mean_p, mean_r


def build_results_table(model_metrics):
    """
    Build a summary DataFrame from a list of metric dicts.

    Parameters
    ----------
    model_metrics : list of dicts with keys: Model, RMSE, MAE,
                    and optionally Precision@K, Recall@K
    """
    return pd.DataFrame(model_metrics).round(4)
