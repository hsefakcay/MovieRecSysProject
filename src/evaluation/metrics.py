"""
Evaluation metrics for recommendation systems.

All metrics are implemented from scratch using NumPy.
Supports:
  - Precision@K
  - Recall@K
  - NDCG@K
  - Mean Average Precision (mAP)
  - AUC (approximation over ranked list)
  - RMSE / MAE (for explicit rating prediction)
"""

import numpy as np
import pandas as pd
from typing import Union


# ---------------------------------------------------------------------------
# Core ranking metrics (single user)
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Precision@K: fraction of top-K recommended items that are relevant.

    Args:
        recommended: ordered list of recommended item indices
        relevant: set of ground-truth relevant item indices
        k: cut-off

    Returns:
        Precision@K in [0, 1]
    """
    if k == 0 or not relevant:
        return 0.0
    top_k = recommended[:k]
    n_hits = sum(1 for item in top_k if item in relevant)
    return n_hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Recall@K: fraction of relevant items that appear in the top-K list.

    Args:
        recommended: ordered list of recommended item indices
        relevant: set of ground-truth relevant item indices
        k: cut-off

    Returns:
        Recall@K in [0, 1]
    """
    if k == 0 or not relevant:
        return 0.0
    top_k = recommended[:k]
    n_hits = sum(1 for item in top_k if item in relevant)
    return n_hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain@K.

    Uses binary relevance: 1 if item in relevant, 0 otherwise.
    IDCG is the DCG of the perfect ranking (all relevant items ranked first).

    Args:
        recommended: ordered list of recommended item indices
        relevant: set of ground-truth relevant item indices
        k: cut-off

    Returns:
        NDCG@K in [0, 1]
    """
    if k == 0 or not relevant:
        return 0.0

    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(rank + 2)   # rank is 0-indexed → log2(rank+2)
        for rank, item in enumerate(top_k)
        if item in relevant
    )

    # Ideal DCG: place all relevant items first (up to K)
    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_len))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(recommended: list, relevant: set) -> float:
    """
    Average Precision (AP) over the full recommended list.

    AP = Σ [P@k × rel(k)] / |relevant|

    Args:
        recommended: ordered list of recommended item indices
        relevant: set of ground-truth relevant item indices

    Returns:
        AP in [0, 1]
    """
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            precision_sum += hits / (rank + 1)

    return precision_sum / len(relevant)


def auc_from_ranked_list(recommended: list, relevant: set, n_items: int) -> float:
    """
    Approximate AUC from a ranked list.

    For each relevant item at position r (1-indexed), count the number of
    non-relevant items ranked above it. AUC = 1 - (avg rank of positives / n_negatives).

    Args:
        recommended: ordered list of recommended item indices (all items or candidates)
        relevant: set of ground-truth relevant item indices
        n_items: total number of items (for AUC denominator)

    Returns:
        Approximate AUC in [0, 1]
    """
    if not relevant or n_items == 0:
        return 0.0

    n_pos = len(relevant)
    n_neg = n_items - n_pos

    if n_neg == 0:
        return 1.0

    # Positions (0-indexed) of relevant items in the ranked list
    pos_ranks = [r for r, item in enumerate(recommended) if item in relevant]

    if not pos_ranks:
        return 0.0

    # AUC via Wilcoxon-Mann-Whitney: fraction of (pos, neg) pairs where pos < neg rank
    n_correct = sum(r - i for i, r in enumerate(sorted(pos_ranks)))
    auc = n_correct / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Rating prediction metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Aggregate evaluation (all users)
# ---------------------------------------------------------------------------

def evaluate_ranking(
    recommendations: dict[int, list],
    test_items: dict[int, set],
    K_list: list[int] = [5, 10, 20],
    n_items: int = None,
) -> pd.DataFrame:
    """
    Evaluate a recommender system across all users.

    Args:
        recommendations: dict of {user_idx: [ranked item_idx list]}
        test_items: dict of {user_idx: set of relevant item_idx}
        K_list: list of K values to evaluate at
        n_items: total item count (for AUC); if None, AUC is skipped

    Returns:
        pd.DataFrame with columns [K, Precision, Recall, NDCG, mAP] (and AUC if n_items given)
        Each row is one K value, averaged over all users.
    """
    users = list(test_items.keys())
    rows = []

    for k in K_list:
        precisions, recalls, ndcgs, aps, aucs = [], [], [], [], []

        for u in users:
            rec = recommendations.get(u, [])
            rel = test_items.get(u, set())
            if not rel:
                continue
            precisions.append(precision_at_k(rec, rel, k))
            recalls.append(recall_at_k(rec, rel, k))
            ndcgs.append(ndcg_at_k(rec, rel, k))
            aps.append(average_precision(rec, rel))
            if n_items is not None:
                aucs.append(auc_from_ranked_list(rec, rel, n_items))

        row = {
            "K": k,
            "Precision@K": np.mean(precisions) if precisions else 0.0,
            "Recall@K":    np.mean(recalls) if recalls else 0.0,
            "NDCG@K":      np.mean(ndcgs) if ndcgs else 0.0,
            "mAP":         np.mean(aps) if aps else 0.0,
        }
        if n_items is not None:
            row["AUC"] = np.mean(aucs) if aucs else 0.0

        rows.append(row)

    return pd.DataFrame(rows).set_index("K")


def evaluate_candidate_recall(
    candidates: dict[int, list],
    test_items: dict[int, set],
    N_list: list[int] = [50, 100, 200],
) -> pd.DataFrame:
    """
    Evaluate Stage 1 candidate generators: Recall@N.

    Args:
        candidates: dict of {user_idx: [candidate item_idx list]}
        test_items: dict of {user_idx: set of relevant item_idx}
        N_list: candidate pool sizes to evaluate

    Returns:
        pd.DataFrame with columns [N, Recall@N]
    """
    users = list(test_items.keys())
    rows = []

    for n in N_list:
        recalls = []
        for u in users:
            cands = candidates.get(u, [])[:n]
            rel = test_items.get(u, set())
            if rel:
                recalls.append(recall_at_k(cands, rel, n))

        rows.append({
            "N": n,
            "Recall@N": np.mean(recalls) if recalls else 0.0,
        })

    return pd.DataFrame(rows).set_index("N")
