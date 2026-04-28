"""
preprocessing.py
Builds the user-item matrix from the cleaned ratings DataFrame.
"""

import pandas as pd
import numpy as np


def build_user_item_matrix(train_df):
    """
    Pivot training ratings into a user x movie matrix.
    Returns the raw matrix (NaN for missing) and the mean-centered version.
    """
    matrix = train_df.pivot_table(
        index='userId', columns='movieId', values='rating'
    )
    user_means      = matrix.mean(axis=1)
    matrix_demeaned = matrix.subtract(user_means, axis=0).fillna(0)
    matrix_filled   = matrix.fillna(0)

    return matrix, matrix_filled, matrix_demeaned, user_means


def get_global_mean(train_df):
    """Return the global mean rating from the training set."""
    return train_df['rating'].mean()
