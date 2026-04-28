"""
cf_model.py
User-based and Item-based Collaborative Filtering using cosine similarity.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_similarity(user_item_filled):
    """Compute user-user cosine similarity matrix."""
    sim = cosine_similarity(user_item_filled)
    return pd.DataFrame(sim, index=user_item_filled.index, columns=user_item_filled.index)


def build_item_similarity(user_item_filled):
    """Compute item-item cosine similarity matrix."""
    sim = cosine_similarity(user_item_filled.T)
    return pd.DataFrame(sim, index=user_item_filled.columns, columns=user_item_filled.columns)


def predict_user_based(user_id, movie_id, user_sim_df, user_item_matrix, global_mean, k=20):
    """
    Predict rating for (user_id, movie_id) using user-based CF.
    Falls back to global mean for cold-start cases.
    """
    if user_id not in user_sim_df.index or movie_id not in user_item_matrix.columns:
        return global_mean

    sim_scores    = user_sim_df[user_id].drop(index=user_id)
    movie_ratings = user_item_matrix[movie_id]
    sim_scores    = sim_scores[movie_ratings > 0]

    if sim_scores.empty or sim_scores.sum() == 0:
        return global_mean

    top_k            = sim_scores.nlargest(k)
    weights          = top_k.values
    neighbor_ratings = user_item_matrix.loc[top_k.index, movie_id].values
    return np.dot(weights, neighbor_ratings) / np.sum(np.abs(weights))


def predict_item_based(user_id, movie_id, item_sim_df, user_item_matrix, global_mean, k=20):
    """
    Predict rating for (user_id, movie_id) using item-based CF.
    Falls back to global mean for cold-start cases.
    """
    if user_id not in user_item_matrix.index or movie_id not in item_sim_df.index:
        return global_mean

    sim_scores   = item_sim_df[movie_id].drop(index=movie_id)
    user_ratings = user_item_matrix.loc[user_id]
    sim_scores   = sim_scores[user_ratings > 0]

    if sim_scores.empty or sim_scores.sum() == 0:
        return global_mean

    top_k            = sim_scores.nlargest(k)
    neighbor_ratings = user_item_matrix.loc[user_id, top_k.index].values
    return np.dot(top_k.values, neighbor_ratings) / np.sum(np.abs(top_k.values))


def tune_k(val_df, predict_fn, sim_df, user_item_matrix, global_mean,
           k_values=(10, 20, 40), extra_kwargs=None):
    """
    Tune the k hyperparameter on a validation sample.
    Returns the best k value.
    """
    from sklearn.metrics import mean_squared_error
    if extra_kwargs is None:
        extra_kwargs = {}

    best_k, best_rmse = k_values[0], float('inf')
    for k in k_values:
        preds = val_df.apply(
            lambda row: predict_fn(
                row['userId'], row['movieId'],
                sim_df, user_item_matrix, global_mean, k=k, **extra_kwargs
            ), axis=1
        )
        rmse = np.sqrt(mean_squared_error(val_df['rating'], preds))
        print(f'  k={k:>3}  RMSE: {rmse:.4f}')
        if rmse < best_rmse:
            best_rmse, best_k = rmse, k

    print(f'Best k: {best_k}')
    return best_k
