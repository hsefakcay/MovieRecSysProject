"""
Stage 1B — Content-Based Filtering (CBF) Candidate Generator.

Uses genre one-hot encodings + cosine similarity between
user genre profiles and item genre vectors.

Role: supplement SVD candidates, improve coverage for genre-rich items.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


class ContentBasedCandidateGenerator:
    """
    Genre-based content-based filtering.

    User profile = weighted average of genre vectors of rated items (weight = rating).
    Item score   = cosine similarity between user profile and item genre vector.

    Usage:
        cbf = ContentBasedCandidateGenerator()
        cbf.fit(item_genre_matrix)
        cbf.build_user_profiles(train_df)
        candidates = cbf.get_candidates(user_idx=0, top_n=100)
    """

    def __init__(self):
        self.item_genre_matrix: np.ndarray | None = None  # (n_items, n_genres)
        self.user_profiles: np.ndarray | None = None       # (n_users, n_genres)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, item_genre_matrix: np.ndarray) -> None:
        """
        Store item genre matrix (precomputed by MovieLensPreprocessor).

        Args:
            item_genre_matrix: np.ndarray of shape (n_items, n_genres)
        """
        # L2-normalize rows for efficient cosine similarity via dot product
        norms = np.linalg.norm(item_genre_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.item_genre_matrix = item_genre_matrix / norms
        print(f"[CBF] Item genre matrix: {self.item_genre_matrix.shape}")

    def build_user_profiles(
        self,
        train_df: pd.DataFrame,
        n_users: int,
        normalize: bool = True,
    ) -> None:
        """
        Build genre profiles for all users from their training ratings.

        Profile = Σ (rating × item_genre_vector) / Σ rating

        Args:
            train_df: training ratings [user_idx, item_idx, rating]
            n_users:  total number of users
            normalize: L2-normalize profiles for cosine similarity
        """
        n_genres = self.item_genre_matrix.shape[1]
        profiles = np.zeros((n_users, n_genres), dtype=np.float32)
        rating_sums = np.zeros(n_users, dtype=np.float32)

        for _, row in train_df.iterrows():
            u  = int(row["user_idx"])
            it = int(row["item_idx"])
            r  = float(row["rating"])
            profiles[u] += r * self.item_genre_matrix[it]  # already L2-normed
            rating_sums[u] += r

        nz = rating_sums > 0
        profiles[nz] /= rating_sums[nz, np.newaxis]

        if normalize:
            norms = np.linalg.norm(profiles, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            profiles /= norms

        self.user_profiles = profiles
        print(f"[CBF] User profiles built: {profiles.shape}")

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def get_candidates(
        self,
        user_idx: int,
        seen_items: set[int] = None,
        top_n: int = 100,
    ) -> list[int]:
        """
        Score items by cosine similarity and return top-N unseen items.

        Args:
            user_idx:   0-indexed user
            seen_items: items to exclude (already seen)
            top_n:      pool size

        Returns:
            Sorted list of item_idx (descending similarity)
        """
        if self.user_profiles is None or self.item_genre_matrix is None:
            raise RuntimeError("Call fit() and build_user_profiles() first.")

        seen = seen_items or set()
        profile = self.user_profiles[user_idx]  # (n_genres,)

        # Cosine scores: dot product (both are L2-normalized)
        scores = self.item_genre_matrix @ profile  # (n_items,)

        # Mask seen items
        scores[list(seen)] = -np.inf

        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return top_indices.tolist()

    def get_all_candidates(
        self,
        user_history: dict[int, set[int]],
        top_n: int = 100,
        verbose: bool = True,
    ) -> dict[int, list[int]]:
        """
        Generate CBF candidates for all users.

        Returns:
            {user_idx: [candidate item_idx]}
        """
        all_users = list(user_history.keys())
        candidates = {}
        iterator = tqdm(all_users, desc=f"[CBF] Generating candidates (top-{top_n})") \
                   if verbose else all_users

        for u in iterator:
            candidates[u] = self.get_candidates(u, seen_items=user_history[u], top_n=top_n)

        return candidates

    # ------------------------------------------------------------------
    # Hybrid union
    # ------------------------------------------------------------------

    @staticmethod
    def union_candidates(
        svd_candidates: dict[int, list[int]],
        cbf_candidates: dict[int, list[int]],
        max_candidates: int = 200,
    ) -> dict[int, list[int]]:
        """
        Merge SVD and CBF candidate lists via union.
        SVD items are ranked first (higher signal), CBF fills remaining slots.

        Args:
            svd_candidates: {user_idx: [item_idx]}
            cbf_candidates: {user_idx: [item_idx]}
            max_candidates: max pool size after union

        Returns:
            {user_idx: [merged item_idx list (up to max_candidates)]}
        """
        merged = {}
        all_users = set(svd_candidates.keys()) | set(cbf_candidates.keys())

        for u in all_users:
            seen_set = set()
            combined = []
            for item in (svd_candidates.get(u, []) + cbf_candidates.get(u, [])):
                if item not in seen_set:
                    seen_set.add(item)
                    combined.append(item)
                if len(combined) >= max_candidates:
                    break
            merged[u] = combined

        return merged
