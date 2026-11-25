"""
Data loader for MovieLens 1M dataset.

Handles:
  - Parsing ratings.dat, movies.dat, users.dat
  - Temporal train/validation/test split (per user, by timestamp)
  - Negative sample generation for NCF training
  - 0-indexed user/item ID remapping
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


class MovieLensLoader:
    """
    Loads and splits the MovieLens 1M dataset.

    Temporal split strategy (per user):
        - Sort each user's ratings by timestamp.
        - Last `test_ratio` fraction  → test set
        - Previous `val_ratio` fraction → validation set
        - Remaining → training set
    This prevents data leakage and simulates a realistic evaluation scenario.
    """

    RATINGS_COLS = ["user_id", "item_id", "rating", "timestamp"]
    MOVIES_COLS  = ["item_id", "title", "genres"]
    USERS_COLS   = ["user_id", "gender", "age", "occupation", "zip"]

    def __init__(
        self,
        data_path: str,
        test_ratio: float = 0.1,
        val_ratio: float  = 0.1,
        min_interactions: int = 5,
        random_seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.test_ratio = test_ratio
        self.val_ratio  = val_ratio
        self.min_interactions = min_interactions
        self.random_seed = random_seed

        # Populated after load
        self.ratings: pd.DataFrame | None = None
        self.movies:  pd.DataFrame | None = None
        self.users:   pd.DataFrame | None = None

        # ID → 0-indexed maps (populated after load_ratings)
        self.user2idx: dict[int, int] = {}
        self.item2idx: dict[int, int] = {}
        self.idx2user: dict[int, int] = {}
        self.idx2item: dict[int, int] = {}

        self.n_users: int = 0
        self.n_items: int = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_ratings(self) -> pd.DataFrame:
        """Parse ratings.dat and remap IDs to 0-indexed integers."""
        path = self.data_path / "ratings.dat"
        df = pd.read_csv(
            path,
            sep="::",
            names=self.RATINGS_COLS,
            engine="python",
            encoding="latin-1",
        )
        # Filter users with too few interactions
        counts = df.groupby("user_id")["item_id"].count()
        valid_users = counts[counts >= self.min_interactions].index
        df = df[df["user_id"].isin(valid_users)].copy()

        # Build 0-indexed mappings
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["item_id"].unique())
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.item2idx = {it: i for i, it in enumerate(unique_items)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {i: it for it, i in self.item2idx.items()}
        self.n_users  = len(unique_users)
        self.n_items  = len(unique_items)

        df["user_idx"] = df["user_id"].map(self.user2idx)
        df["item_idx"] = df["item_id"].map(self.item2idx)

        self.ratings = df.reset_index(drop=True)
        return self.ratings

    def load_movies(self) -> pd.DataFrame:
        """Parse movies.dat with genre list as a Python list."""
        path = self.data_path / "movies.dat"
        df = pd.read_csv(
            path,
            sep="::",
            names=self.MOVIES_COLS,
            engine="python",
            encoding="latin-1",
        )
        df["genre_list"] = df["genres"].str.split("|")
        self.movies = df
        return self.movies

    def load_users(self) -> pd.DataFrame:
        """Parse users.dat."""
        path = self.data_path / "users.dat"
        df = pd.read_csv(
            path,
            sep="::",
            names=self.USERS_COLS,
            engine="python",
            encoding="latin-1",
        )
        self.users = df
        return self.users

    def load_all(self) -> tuple:
        """Convenience wrapper — load ratings, movies, and users."""
        self.load_ratings()
        self.load_movies()
        self.load_users()
        return self.ratings, self.movies, self.users

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def temporal_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporal train / val / test split **per user**.

        Returns:
            train_df, val_df, test_df  (all with user_idx / item_idx columns)
        """
        if self.ratings is None:
            self.load_ratings()

        df = self.ratings.sort_values(["user_idx", "timestamp"])

        train_rows, val_rows, test_rows = [], [], []

        for _, group in df.groupby("user_idx"):
            n = len(group)
            n_test = max(1, int(np.floor(n * self.test_ratio)))
            n_val  = max(1, int(np.floor(n * self.val_ratio)))
            n_train = n - n_test - n_val

            if n_train < 1:
                # Too few ratings — put everything in train
                train_rows.append(group)
                continue

            train_rows.append(group.iloc[:n_train])
            val_rows.append(group.iloc[n_train: n_train + n_val])
            test_rows.append(group.iloc[n_train + n_val:])

        train_df = pd.concat(train_rows).reset_index(drop=True)
        val_df   = pd.concat(val_rows).reset_index(drop=True)
        test_df  = pd.concat(test_rows).reset_index(drop=True)

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Negative sampling (for NCF)
    # ------------------------------------------------------------------

    def generate_negatives(
        self,
        train_df: pd.DataFrame,
        n_neg: int = 4,
    ) -> pd.DataFrame:
        """
        For each positive (user, item) pair in train_df, sample `n_neg`
        negative items that the user has NOT interacted with.

        Returns a DataFrame with columns: user_idx, item_idx, label (0/1)
        """
        rng = np.random.default_rng(self.random_seed)

        # Build set of observed items per user (for fast exclusion)
        user_items: dict[int, set[int]] = (
            train_df.groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

        all_items = np.arange(self.n_items)
        rows = []

        for _, row in train_df.iterrows():
            u = int(row["user_idx"])
            pos_item = int(row["item_idx"])
            rows.append((u, pos_item, 1))

            seen = user_items.get(u, set())
            neg_pool = np.setdiff1d(all_items, list(seen), assume_unique=True)

            neg_samples = rng.choice(
                neg_pool,
                size=min(n_neg, len(neg_pool)),
                replace=False,
            )
            for neg_item in neg_samples:
                rows.append((u, int(neg_item), 0))

        return pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_user_history(self, train_df: pd.DataFrame) -> dict[int, set[int]]:
        """Return dict: user_idx → set of item_idx seen in train_df."""
        return train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    def get_test_items(self, test_df: pd.DataFrame) -> dict[int, set[int]]:
        """Return dict: user_idx → set of item_idx in test set (ground truth)."""
        return test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    def summary(self) -> None:
        """Print dataset statistics."""
        if self.ratings is None:
            print("Ratings not loaded yet. Call load_ratings() first.")
            return
        n_ratings = len(self.ratings)
        sparsity = 1 - n_ratings / (self.n_users * self.n_items)
        print(f"{'='*45}")
        print(f"  MovieLens 1M — Dataset Summary")
        print(f"{'='*45}")
        print(f"  Users  : {self.n_users:>8,}")
        print(f"  Items  : {self.n_items:>8,}")
        print(f"  Ratings: {n_ratings:>8,}")
        print(f"  Sparsity: {sparsity:.2%}")
        print(f"{'='*45}")
