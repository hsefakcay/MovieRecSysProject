"""
Preprocessing utilities for MovieLens 1M.

Handles:
  - Genre one-hot encoding (18 genres)
  - User genre profile construction (from training ratings)
  - Sparse interaction matrix building
  - Rating normalization helpers
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# All 18 genres present in MovieLens 1M
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


class MovieLensPreprocessor:
    """
    Feature engineering and encoding for MovieLens 1M.
    """

    def __init__(self, genres: list[str] = None):
        self.genres = genres or ALL_GENRES
        self.genre2idx = {g: i for i, g in enumerate(self.genres)}

    # ------------------------------------------------------------------
    # Item features
    # ------------------------------------------------------------------

    def genre_one_hot(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame of shape (n_items, n_genres) with one-hot genre vectors.
        Index = item_id (original MovieLens ID).
        """
        genre_matrix = np.zeros((len(movies_df), len(self.genres)), dtype=np.float32)

        for i, row in movies_df.iterrows():
            for g in row["genre_list"]:
                if g in self.genre2idx:
                    genre_matrix[i, self.genre2idx[g]] = 1.0

        df = pd.DataFrame(
            genre_matrix,
            index=movies_df["item_id"].values,
            columns=self.genres,
        )
        return df

    def genre_one_hot_indexed(
        self,
        movies_df: pd.DataFrame,
        item2idx: dict[int, int],
    ) -> np.ndarray:
        """
        Return a numpy array of shape (n_items, n_genres) ordered by 0-indexed item_idx.
        Items not in item2idx are skipped.
        """
        n_items = len(item2idx)
        matrix = np.zeros((n_items, len(self.genres)), dtype=np.float32)

        for _, row in movies_df.iterrows():
            if row["item_id"] not in item2idx:
                continue
            idx = item2idx[row["item_id"]]
            for g in row["genre_list"]:
                if g in self.genre2idx:
                    matrix[idx, self.genre2idx[g]] = 1.0

        return matrix  # shape: (n_items, 18)

    # ------------------------------------------------------------------
    # User genre profiles
    # ------------------------------------------------------------------

    def build_user_genre_profiles(
        self,
        train_df: pd.DataFrame,
        item_genre_matrix: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute user genre profiles as weighted averages of item genre vectors,
        weighted by the user's ratings.

        Args:
            train_df: training ratings with columns [user_idx, item_idx, rating]
            item_genre_matrix: shape (n_items, n_genres) from genre_one_hot_indexed()
            normalize: if True, L2-normalize each user profile

        Returns:
            user_profiles: np.ndarray of shape (n_users, n_genres)
        """
        n_users = train_df["user_idx"].max() + 1
        n_genres = item_genre_matrix.shape[1]
        profiles = np.zeros((n_users, n_genres), dtype=np.float32)
        rating_sums = np.zeros(n_users, dtype=np.float32)

        for _, row in train_df.iterrows():
            u = int(row["user_idx"])
            it = int(row["item_idx"])
            r = float(row["rating"])
            profiles[u] += r * item_genre_matrix[it]
            rating_sums[u] += r

        # Avoid division by zero
        nz = rating_sums > 0
        profiles[nz] = profiles[nz] / rating_sums[nz, np.newaxis]

        if normalize:
            norms = np.linalg.norm(profiles, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            profiles = profiles / norms

        return profiles  # shape: (n_users, n_genres)

    # ------------------------------------------------------------------
    # Interaction matrix
    # ------------------------------------------------------------------

    def build_interaction_matrix(
        self,
        train_df: pd.DataFrame,
        n_users: int,
        n_items: int,
        binary: bool = False,
    ) -> csr_matrix:
        """
        Build a sparse user-item interaction matrix.

        Args:
            binary: if True, store 1 for any interaction (implicit); 
                    if False, store actual rating value.
        """
        values = np.ones(len(train_df), dtype=np.float32) if binary \
                 else train_df["rating"].values.astype(np.float32)

        matrix = csr_matrix(
            (values, (train_df["user_idx"].values, train_df["item_idx"].values)),
            shape=(n_users, n_items),
        )
        return matrix

    # ------------------------------------------------------------------
    # Rating normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_ratings(df: pd.DataFrame, min_r: float = 1.0, max_r: float = 5.0) -> pd.DataFrame:
        """Scale ratings to [0, 1]."""
        df = df.copy()
        df["rating"] = (df["rating"] - min_r) / (max_r - min_r)
        return df

    @staticmethod
    def binarize_ratings(df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
        """Convert ratings ≥ threshold to 1 (liked), else 0 (not liked)."""
        df = df.copy()
        df["rating"] = (df["rating"] >= threshold).astype(np.float32)
        return df
