"""
data_loader.py
Loads and cleans the CiaoDVD dataset from raw .txt files.
"""

import os
import pandas as pd


RAW_DATA_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def load_ratings(filepath=None):
    """Load and clean the ratings file. Returns a cleaned DataFrame."""
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, 'ratings.txt')

    ratings = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        names=['userId', 'movieId', 'rating']
    )

    # Drop duplicates, missing values, and out-of-range ratings
    before = len(ratings)
    ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
    ratings = ratings.dropna(subset=['rating'])
    ratings = ratings[ratings['rating'].between(1, 5)]
    ratings = ratings[['userId', 'movieId', 'rating']].reset_index(drop=True)

    print(f'Ratings loaded   : {before:,}')
    print(f'Ratings after cleaning : {len(ratings):,}')
    return ratings


def load_trust(filepath=None):
    """Load the trust network file. Returns a DataFrame."""
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, 'trust.txt')

    trust = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        names=['trustorId', 'trusteeId', 'trustRating']
    )
    print(f'Trust links loaded: {len(trust):,}')
    return trust


def save_clean_ratings(ratings, filepath=None):
    """Save cleaned ratings to processed directory."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    if filepath is None:
        filepath = os.path.join(PROCESSED_DATA_DIR, 'ratings_clean.csv')
    ratings.to_csv(filepath, index=False)
    print(f'Saved cleaned ratings to {filepath}')


if __name__ == '__main__':
    ratings = load_ratings()
    trust   = load_trust()
    save_clean_ratings(ratings)
