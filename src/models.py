from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
if TYPE_CHECKING:
    from typing import Self


# ---------------------------------------------------------------------------
# Supervised baselines
# ---------------------------------------------------------------------------


class LogisticRegressionModel:
    """Logistic Regression wrapper following the common model interface."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        seed: int = 0,
    ) -> None: 
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.seed = seed
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.seed
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> Self:
        """Train the LR model."""
        self.model.fit(X_train, y_train)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability (higher = more anomalous)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray: 
        """Return binary predictions based on threshold."""
        return (self.score_samples(X) >= threshold).astype(int)
     
class XGBoostModel:
    """XGBoost wrapper following the common model interface."""

    def __init__(self, seed: int = 0, **kwargs) -> None: 
        self.seed = seed
        self.kwargs = kwargs
        self.model = XGBClassifier(
            random_state=self.seed,
            eval_metric="aucpr",
            early_stopping_rounds=50,
            verbosity=0,
            **self.kwargs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Self:
        """Train the XGBoost model with early stopping on val set."""
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability (higher = more anomalous)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary predictions based on threshold."""
        return (self.score_samples(X) >= threshold).astype(int)

# ---------------------------------------------------------------------------
# Classical unsupervised
# ---------------------------------------------------------------------------


class IsolationForestModel:
    """Isolation Forest wrapper following the common model interface."""

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: str | float = "auto",
        max_samples: str | int = "auto",
        seed: int = 0,
    ) -> None: 
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.seed = seed
        self.model = IsolationForest(
        n_estimators = self.n_estimators,
        contamination = self.contamination,
        max_samples = self.max_samples,
        random_state = self.seed
    ) 
    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> Self: 
        """Train the Isolation Forest model."""
        self.model.fit(X_train)
        return self
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return negated sklearn anomaly score (higher = more anomalous)."""
        return -self.model.score_samples(X)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary predictions based on threshold."""
        return (self.score_samples(X) >= threshold).astype(int)

class LOFModel:
    """Local Outlier Factor wrapper (novelty=True) following the common model interface."""

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: str | float = "auto",
        novelty: bool = True
    ) -> None: 
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty
        self.model = LocalOutlierFactor(
            n_neighbors= self.n_neighbors,
            contamination= self.contamination,
            novelty= self.novelty
        )
        

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> Self: 
        """Train the LOF model."""
        self.model.fit(X_train)
        return self
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return negated LOF score (higher = more anomalous)."""
        return -self.model.decision_function(X)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray: 
        """Return binary predictions based on threshold."""
        return (self.score_samples(X) >= threshold).astype(int)


class OneClassSVMModel:
    """One-Class SVM wrapper following the common model interface."""

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.1,
        gamma: str | float = "auto",
        seed: int = 0,
    ) -> None:
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.seed = seed  # OC-SVM is deterministic; accepted for interface uniformity
        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> Self: 
        """Train the One-Class SVM model."""
        self.model.fit(X_train)
        return self
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return negated decision function (higher = more anomalous)."""
        return -self.model.decision_function(X)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary predictions based on threshold."""
        return (self.score_samples(X) >= threshold).astype(int)

# ---------------------------------------------------------------------------
# Deep reconstruction — Autoencoder
# ---------------------------------------------------------------------------


class _AutoencoderNet(nn.Module):
    """Symmetric encoder-decoder MLP."""
    
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        # Encoder — no activation/dropout on the last (bottleneck) layer
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            if i < len(hidden_dims) - 1:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder mirrors encoder — no activation on the output (data is standardized)
        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        for i, h_dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            if i < len(reversed_dims) - 1:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstructed input."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class AutoencoderModel:
    """Autoencoder anomaly detector: score = mean squared reconstruction error."""

    def __init__(
        self,
        hidden_dims: list[int],
        latent_dim: int,
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        dropout: float = 0.1,
        seed: int = 0,
        device: str = "cpu",
    ) -> None: 
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.seed = seed
        self.device = torch.device(device)
        self.net: _AutoencoderNet | None = None
        self.train_losses: list[float] = []

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> Self:
        """Train the autoencoder on normal samples only (X_train should be pre-filtered).

        X_val is unused here but accepted for interface consistency.
        """
        torch.manual_seed(self.seed)
        input_dim = X_train.shape[1]
        full_dims = self.hidden_dims + [self.latent_dim]
        self.net = _AutoencoderNet(input_dim, full_dims, self.dropout).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.net.train()
        self.train_losses = []
        for _ in range(self.epochs):
            epochs_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                optimizer.step()
                epochs_loss += loss.item()
            self.train_losses.append(epochs_loss / len(loader))

        self.net.eval()
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return mean squared reconstruction error per sample (higher = more anomalous)."""
        assert self.net is not None, "Call fit() before score_samples()"
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            x_hat = self.net(X_t)
            mse = ((X_t - x_hat) ** 2).mean(dim=1)
        return mse.cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary predictions based on threshold."""
        return (self.score_samples(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Deep reconstruction — VAE
# ---------------------------------------------------------------------------


class _VAENet(nn.Module):
    """Encoder-decoder MLP with reparameterisation trick."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None: 
        super().__init__()
        encoder_layers: list[nn.Module] = []
        prev_dim  =input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(prev_dim,latent_dim)
        self.log_var = nn.Linear(prev_dim, latent_dim)

        decoder_layers : list[nn.Module] = []
        prev_dim = latent_dim 
        reversed_dims = list(reversed(hidden_dims))
        for h_dim in reversed_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
        """Return (mu, log_var) of latent distribution."""
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)
    
    def reparameterize(self,mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Return sampled latent vector z."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps *std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor: 
        """Return reconstructed input."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_recon, mu, log_var)."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return (x_recon, mu, log_var)

class VAEModel:
    """Variational Autoencoder anomaly detector: score = reconstruction loss."""

    def __init__(
        self,
        hidden_dims: list[int],
        latent_dim: int,
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        beta: float = 1.0,
        seed: int = 0,
        device: str = "cpu",
    ) -> None: 
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta = beta
        self.seed = seed
        self.device = torch.device(device)
        self.net: _VAENet | None = None
        self.train_losses: list[float] = []


    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> Self:
        """Train the VAE on normal samples only (X_train should be pre-filtered)."""
        torch.manual_seed(self.seed)
        input_dim = X_train.shape[1]
        self.net = _VAENet(input_dim, self.hidden_dims,self.latent_dim).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.net.train()
        self.train_losses= []
        for _ in range(self.epochs):
            epochs_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                x_recon, mu, log_var = self.net(batch)
                recon_loss = F.mse_loss(x_recon, batch, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2)- log_var.exp())
                loss = recon_loss + self.beta * kl_loss
                loss.backward()
                optimizer.step()
                epochs_loss += loss.item()
            self.train_losses.append(epochs_loss / len(loader))

        self.net.eval()
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample reconstruction loss (higher = more anomalous)."""
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            X_hat, _, _  = self.net(X_t)
            errors = ((X_t - X_hat) ** 2).mean(dim=1) 
        return errors.cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary predictions based on threshold."""
        return (self.score_samples(X) >= threshold).astype(int)
# ---------------------------------------------------------------------------
# Deep one-class — Deep SVDD  (Ruff et al., ICML 2018)
# ---------------------------------------------------------------------------


class _SVDDNet(nn.Module):
    """MLP mapping inputs to a compact representation space."""

    def __init__(self, input_dim: int, hidden_dims: list[int], rep_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim, bias=False))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, rep_dim, bias=False))
        self.net = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x)

class DeepSVDDModel:
    """Deep SVDD anomaly detector.

    Score = distance from the learned hypersphere center c.
    Center c is initialised as the mean of network outputs on normal training data.
    """

    def __init__(
        self,
        hidden_dims: list[int],
        rep_dim: int,
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        weight_decay: float = 1e-6,
        seed: int = 0,
        device: str = "cpu",
    ) -> None: 
        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = torch.device(device)
        self.net: _SVDDNet | None = None
        self.c: torch.Tensor | None = None
        self.train_losses: list[float] = []

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> Self:
        """Train Deep SVDD on normal samples only (X_train should be pre-filtered)."""
        torch.manual_seed(self.seed)
        input_dim = X_train.shape[1]
        self.net = _SVDDNet(input_dim, self.hidden_dims, self.rep_dim).to(self.device)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Initialise center c as mean of network outputs on normal data (Ruff et al.)
        self.net.eval()
        with torch.no_grad():
            z_all = self.net(X_tensor.to(self.device))
            c = z_all.mean(dim=0)
            # Avoid c coordinates too close to zero (trivial solution mapping everything to 0)
            eps = 0.1
            c[(torch.abs(c) < eps) & (c < 0)] = -eps
            c[(torch.abs(c) < eps) & (c >= 0)] = eps
        self.c = c.detach()

        self.net.train()
        self.train_losses= []
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                z = self.net(batch)
                loss = ((z- self.c)**2).sum(dim=1).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.train_losses.append(epoch_loss / len(loader))
        self.net.eval()
        return self



    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return distance to hypersphere center c (higher = more anomalous)."""
        self.net.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z = self.net(X_tensor)
            scores = ((z- self.c)**2).sum(dim=1)
        return scores.cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary predictions based on threshold.""" 
        return (self.score_samples(X) > threshold).astype(int)
