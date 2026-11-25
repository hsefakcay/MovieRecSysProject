"""
Stage 2 — Neural Collaborative Filtering (NeuMF).

Based on:
    He et al., "Neural Collaborative Filtering," WWW 2017.
    https://dl.acm.org/doi/10.1145/3038912.3052569

Architecture:
    Two parallel branches:
      • GMF (Generalized Matrix Factorization) — element-wise product of embeddings
      • MLP — concatenated embeddings through fully-connected layers
    Final layer: concatenate GMF output + MLP output → FC(1) → Sigmoid

Training:
    - Binary Cross-Entropy loss on implicit feedback
    - Negative sampling (n_neg negatives per positive)
    - Adam optimizer
    - Device: MPS (Apple Silicon) > CUDA > CPU

Usage:
    model = NeuMF(n_users, n_items)
    trainer = NCFTrainer(model, device="mps")
    trainer.train(train_df, loader, n_epochs=20)
    recs = trainer.recommend_from_candidates(user_idx, candidates, k=10)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Device selection (MPS → CUDA → CPU)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class NCFDataset(Dataset):
    """Dataset wrapping (user_idx, item_idx, label) triples."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with columns [user_idx, item_idx, label]
        """
        self.users  = torch.LongTensor(df["user_idx"].values)
        self.items  = torch.LongTensor(df["item_idx"].values)
        self.labels = torch.FloatTensor(df["label"].values)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# NeuMF Model
# ---------------------------------------------------------------------------

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization model (NeuMF / NCF).

    Combines:
      - GMF: element-wise product of MF embeddings (linear interactions)
      - MLP: concatenated MLP embeddings through non-linear layers

    Args:
        n_users:    total number of users
        n_items:    total number of items
        mf_dim:     embedding dimension for the GMF branch
        mlp_layers: hidden sizes for the MLP branch [input → … → output]
                    The MLP embedding dim = mlp_layers[0] // 2
        dropout:    dropout probability applied after each MLP hidden layer
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        mf_dim: int = 64,
        mlp_layers: list[int] = [256, 128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()

        # GMF branch embeddings
        self.mf_user_emb = nn.Embedding(n_users, mf_dim)
        self.mf_item_emb = nn.Embedding(n_items, mf_dim)

        # MLP branch embeddings (input dim = mlp_layers[0])
        mlp_emb_dim = mlp_layers[0] // 2
        self.mlp_user_emb = nn.Embedding(n_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, mlp_emb_dim)

        # MLP layers
        mlp_modules = []
        input_size = mlp_layers[0]
        for output_size in mlp_layers[1:]:
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_size = output_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Final prediction layer: GMF (mf_dim) + MLP last hidden (mlp_layers[-1]) → 1
        self.predict_layer = nn.Linear(mf_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for embeddings and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_ids: torch.LongTensor,
        item_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            user_ids: (batch_size,) user indices
            item_ids: (batch_size,) item indices

        Returns:
            (batch_size,) click probability in [0, 1]
        """
        # GMF branch
        gmf_u = self.mf_user_emb(user_ids)   # (B, mf_dim)
        gmf_i = self.mf_item_emb(item_ids)   # (B, mf_dim)
        gmf_out = gmf_u * gmf_i              # element-wise product

        # MLP branch
        mlp_u = self.mlp_user_emb(user_ids)  # (B, mlp_emb_dim)
        mlp_i = self.mlp_item_emb(item_ids)  # (B, mlp_emb_dim)
        mlp_input = torch.cat([mlp_u, mlp_i], dim=-1)  # (B, mlp_emb_dim*2)
        mlp_out = self.mlp(mlp_input)        # (B, mlp_layers[-1])

        # Fusion
        fusion = torch.cat([gmf_out, mlp_out], dim=-1)  # (B, mf_dim + mlp_layers[-1])
        logit = self.predict_layer(fusion).squeeze(-1)   # (B,)
        return self.sigmoid(logit)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NCFTrainer:
    """
    Training and inference wrapper for NeuMF.

    Usage:
        trainer = NCFTrainer(model)
        trainer.train(train_neg_df, n_epochs=20, batch_size=1024)
        recs = trainer.recommend_from_candidates(user_idx, candidates, k=10)
    """

    def __init__(
        self,
        model: NeuMF,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: torch.device = None,
    ):
        self.device = device or get_device()
        self.model  = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
        self.train_losses: list[float] = []

        print(f"[NeuMF] Device: {self.device}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[NeuMF] Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_neg_df: pd.DataFrame,
        n_epochs: int = 20,
        batch_size: int = 1024,
        num_workers: int = 0,
    ) -> None:
        """
        Train NeuMF on the negative-sampled training set.

        Args:
            train_neg_df: DataFrame with [user_idx, item_idx, label (0/1)]
            n_epochs:     number of training epochs
            batch_size:   mini-batch size
            num_workers:  DataLoader workers (0 = main process)
        """
        dataset = NCFDataset(train_neg_df)
        loader  = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.model.train()
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            for users, items, labels in loader:
                users  = users.to(self.device)
                items  = items.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(users, items)
                loss  = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_loss)
            print(f"  Epoch {epoch:>3}/{n_epochs} | Loss: {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_candidates(
        self,
        user_idx: int,
        candidates: list[int],
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Score a list of candidate items for a single user.

        Returns:
            np.ndarray of shape (len(candidates),) with click probabilities
        """
        self.model.eval()
        all_scores = []

        for start in range(0, len(candidates), batch_size):
            batch_items = candidates[start: start + batch_size]
            users = torch.LongTensor([user_idx] * len(batch_items)).to(self.device)
            items = torch.LongTensor(batch_items).to(self.device)
            scores = self.model(users, items).cpu().numpy()
            all_scores.append(scores)

        return np.concatenate(all_scores)

    def recommend_from_candidates(
        self,
        user_idx: int,
        candidates: list[int],
        k: int = 10,
    ) -> list[int]:
        """
        Re-rank candidates and return top-K recommendations.

        Args:
            user_idx:   user to recommend for
            candidates: pre-filtered candidate item indices
            k:          final list size

        Returns:
            List of top-K item_idx sorted by predicted click probability
        """
        if not candidates:
            return []
        scores = self.score_candidates(user_idx, candidates)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:k]]

    def recommend_all(
        self,
        candidates_per_user: dict[int, list[int]],
        k: int = 10,
        verbose: bool = True,
    ) -> dict[int, list[int]]:
        """
        Generate top-K recommendations for all users from pre-generated candidates.

        Args:
            candidates_per_user: {user_idx: [candidate item_idx]}
            k: final list size

        Returns:
            {user_idx: [top-K item_idx]}
        """
        recommendations = {}
        users = list(candidates_per_user.keys())
        iterator = tqdm(users, desc=f"[NeuMF] Scoring candidates (K={k})") \
                   if verbose else users

        for u in iterator:
            recommendations[u] = self.recommend_from_candidates(
                u, candidates_per_user[u], k=k
            )

        return recommendations

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), path)
        print(f"[NeuMF] Model saved → {path}")

    def load(self, path: str) -> None:
        """Load model weights from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"[NeuMF] Model loaded ← {path}")
