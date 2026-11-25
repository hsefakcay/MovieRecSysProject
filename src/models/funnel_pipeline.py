"""
Two-Stage Funnel Pipeline Orchestrator.

Integrates:
  Stage 1: SVD + CBF candidate generation
  Stage 2: NeuMF scoring and re-ranking

Usage:
    pipeline = TwoStageFunnelPipeline(svd_gen, cbf_gen, ncf_trainer)
    pipeline.fit(train_df, train_neg_df, movies_df, loader)
    results  = pipeline.evaluate(test_items, user_history, K_list=[5, 10, 20])
"""

import numpy as np
import pandas as pd

from src.data.loader import MovieLensLoader
from src.data.preprocessor import MovieLensPreprocessor
from src.models.svd_model import SVDCandidateGenerator
from src.models.cbf_model import ContentBasedCandidateGenerator
from src.models.ncf_model import NeuMF, NCFTrainer, get_device
from src.evaluation.metrics import evaluate_ranking, evaluate_candidate_recall


class TwoStageFunnelPipeline:
    """
    End-to-end two-stage recommendation funnel.

    Stage 1: Retrieve → SVD + CBF candidate pool (Top-N per user)
    Stage 2: Rank    → NeuMF re-ranks candidates (Top-K final list)

    Args:
        svd_gen:         SVDCandidateGenerator instance
        cbf_gen:         ContentBasedCandidateGenerator instance
        ncf_trainer:     NCFTrainer instance
        top_n:           Stage 1 candidate pool size (per model)
        max_candidates:  Max size of merged Stage 1 pool (union)
        top_k:           Final recommendation list size
    """

    def __init__(
        self,
        svd_gen: SVDCandidateGenerator,
        cbf_gen: ContentBasedCandidateGenerator,
        ncf_trainer: NCFTrainer,
        top_n: int = 100,
        max_candidates: int = 200,
        top_k: int = 10,
    ):
        self.svd_gen       = svd_gen
        self.cbf_gen       = cbf_gen
        self.ncf_trainer   = ncf_trainer
        self.top_n         = top_n
        self.max_candidates = max_candidates
        self.top_k         = top_k

        # Populated after fit()
        self.svd_candidates: dict[int, list[int]] = {}
        self.cbf_candidates: dict[int, list[int]] = {}
        self.merged_candidates: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # Candidate generation (Stage 1)
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        user_history: dict[int, set[int]],
        verbose: bool = True,
    ) -> dict[int, list[int]]:
        """
        Run Stage 1: generate merged candidate pool for all users.

        Args:
            user_history: {user_idx: set of seen item_idx}

        Returns:
            {user_idx: [merged candidate item_idx list]}
        """
        if verbose:
            print("\n[Funnel] Stage 1 — Candidate Generation")

        self.svd_candidates = self.svd_gen.get_all_candidates(
            user_history, top_n=self.top_n, verbose=verbose
        )
        self.cbf_candidates = self.cbf_gen.get_all_candidates(
            user_history, top_n=self.top_n, verbose=verbose
        )
        self.merged_candidates = ContentBasedCandidateGenerator.union_candidates(
            self.svd_candidates,
            self.cbf_candidates,
            max_candidates=self.max_candidates,
        )

        if verbose:
            avg_pool = np.mean([len(v) for v in self.merged_candidates.values()])
            print(f"[Funnel] Average candidate pool size: {avg_pool:.1f}")

        return self.merged_candidates

    # ------------------------------------------------------------------
    # Ranking (Stage 2)
    # ------------------------------------------------------------------

    def rank_candidates(
        self,
        candidates: dict[int, list[int]] = None,
        k: int = None,
        verbose: bool = True,
    ) -> dict[int, list[int]]:
        """
        Run Stage 2: re-rank candidates with NeuMF and return top-K lists.

        Args:
            candidates: merged candidate dict (uses self.merged_candidates if None)
            k: final list size (uses self.top_k if None)

        Returns:
            {user_idx: [top-K item_idx]}
        """
        candidates = candidates or self.merged_candidates
        k = k or self.top_k

        if verbose:
            print("\n[Funnel] Stage 2 — NeuMF Ranking")

        return self.ncf_trainer.recommend_all(candidates, k=k, verbose=verbose)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        user_history: dict[int, set[int]],
        k: int = None,
        verbose: bool = True,
    ) -> dict[int, list[int]]:
        """
        Execute the full two-stage funnel for all users.

        Returns:
            {user_idx: [top-K recommended item_idx]}
        """
        candidates    = self.generate_candidates(user_history, verbose=verbose)
        recommendations = self.rank_candidates(candidates, k=k, verbose=verbose)
        return recommendations

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_stage1(
        self,
        test_items: dict[int, set[int]],
        N_list: list[int] = None,
    ) -> pd.DataFrame:
        """
        Evaluate Stage 1 candidate coverage (Recall@N).

        Returns:
            pd.DataFrame with Recall@N for SVD, CBF, and merged candidates.
        """
        N_list = N_list or [50, 100, 200]

        from src.evaluation.metrics import evaluate_candidate_recall

        svd_results    = evaluate_candidate_recall(self.svd_candidates, test_items, N_list)
        cbf_results    = evaluate_candidate_recall(self.cbf_candidates, test_items, N_list)
        merged_results = evaluate_candidate_recall(self.merged_candidates, test_items, N_list)

        combined = pd.DataFrame({
            "SVD Recall@N":    svd_results["Recall@N"],
            "CBF Recall@N":    cbf_results["Recall@N"],
            "Merged Recall@N": merged_results["Recall@N"],
        })
        return combined

    def evaluate_stage2(
        self,
        recommendations: dict[int, list[int]],
        test_items: dict[int, set[int]],
        K_list: list[int] = None,
        n_items: int = None,
    ) -> pd.DataFrame:
        """
        Evaluate Stage 2 ranking quality (Precision, Recall, NDCG, mAP, AUC).

        Returns:
            pd.DataFrame indexed by K.
        """
        K_list = K_list or [5, 10, 20]
        return evaluate_ranking(recommendations, test_items, K_list, n_items)
