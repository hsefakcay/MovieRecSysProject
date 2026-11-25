"""
Stage 1A — SVD-based Candidate Generator.

Uses scikit-surprise's SVD (Simon Funk SVD with biases).
Role: produce a Top-N candidate pool per user for Stage 2 scoring.
Key metric: Recall@N — fraction of test items that appear in the candidate pool.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from surprise import SVD, Dataset, Reader
    from surprise import accuracy as surprise_accuracy
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("[SVDModel] scikit-surprise not found. Install with: pip install scikit-surprise")


class SVDCandidateGenerator:
    """
    Matrix Factorization via SVD (scikit-surprise).

    Usage:
        gen = SVDCandidateGenerator(n_factors=100, n_epochs=20)
        gen.fit(train_df)
        candidates = gen.get_candidates(user_idx=0, top_n=100)
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int  = 20,
        lr_all: float  = 0.005,
        reg_all: float = 0.02,
        random_seed: int = 42,
    ):
        if not SURPRISE_AVAILABLE:
            raise ImportError("scikit-surprise is required. Run: pip install scikit-surprise")

        self.n_factors   = n_factors
        self.n_epochs    = n_epochs
        self.lr_all      = lr_all
        self.reg_all     = reg_all
        self.random_seed = random_seed

        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_seed,
            verbose=False,
        )

        # Maps from 0-indexed back to original IDs (set during fit)
        self.idx2user: dict[int, int] = {}
        self.idx2item: dict[int, int] = {}
        self._n_items: int = 0
        self._all_items: np.ndarray = np.array([])
        self._trainset = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        idx2user: dict[int, int] = None,
        idx2item: dict[int, int] = None,
    ) -> None:
        """
        Train SVD on the training ratings.

        Args:
            train_df: DataFrame with columns [user_idx, item_idx, rating]
            idx2user: mapping from 0-idx → original user_id (optional)
            idx2item: mapping from 0-idx → original item_id (optional)
        """
        self.idx2user = idx2user or {}
        self.idx2item = idx2item or {}
        self._n_items = train_df["item_idx"].nunique()
        self._all_items = train_df["item_idx"].unique()

        # surprise requires (user, item, rating) with original IDs or string IDs
        # We use 0-indexed integers directly — surprise handles them fine
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_df[["user_idx", "item_idx", "rating"]],
            reader,
        )
        self._trainset = data.build_full_trainset()
        self.model.fit(self._trainset)
        print(f"[SVD] Trained — {self._n_items} items, {train_df['user_idx'].nunique()} users")

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
        Predict ratings for all unseen items and return top-N as candidates.

        Args:
            user_idx:   0-indexed user
            seen_items: items already seen by the user (excluded from candidates)
            top_n:      candidate pool size

        Returns:
            List of item_idx values sorted by predicted rating (descending)
        """
        seen = seen_items or set()
        candidate_items = [i for i in self._all_items if i not in seen]

        predictions = [(item, self.model.predict(user_idx, item).est)
                       for item in candidate_items]
        predictions.sort(key=lambda x: x[1], reverse=True)

        return [item for item, _ in predictions[:top_n]]

    def get_all_candidates(
        self,
        user_history: dict[int, set[int]],
        top_n: int = 100,
        verbose: bool = True,
    ) -> dict[int, list[int]]:
        """
        Generate candidate lists for all users.

        Args:
            user_history: {user_idx: set of seen item_idx}
            top_n: pool size per user
            verbose: show progress bar

        Returns:
            {user_idx: [candidate item_idx]}
        """
        all_users = list(user_history.keys())
        candidates = {}
        iterator = tqdm(all_users, desc=f"[SVD] Generating candidates (top-{top_n})") \
                   if verbose else all_users

        for u in iterator:
            candidates[u] = self.get_candidates(u, seen_items=user_history[u], top_n=top_n)

        return candidates

    # ------------------------------------------------------------------
    # Rating prediction (for RMSE / MAE)
    # ------------------------------------------------------------------

    def predict_ratings(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every row in test_df."""
        preds = [self.model.predict(int(r["user_idx"]), int(r["item_idx"])).est
                 for _, r in test_df.iterrows()]
        return np.array(preds, dtype=np.float32)
