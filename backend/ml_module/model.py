
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import trapezoid
class _ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class SVClassifierCNN(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            _ResBlock(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            _ResBlock(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            _ResBlock(256),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.gap(x)
        return self.head(x).squeeze(1)

class SVModel:
    def __init__(
        self,
        input_dim: Optional[int] = None,
        dropout: float = 0.3,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dropout = dropout
        self.input_dim: Optional[int] = input_dim
        self.model: Optional[SVClassifierCNN] = None
        self.training_pairs: set = set()
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []
        }
        self.optimizer_state: Optional[Dict] = None

        if input_dim is not None:
            self._build_model(input_dim)

    def _build_model(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.model = SVClassifierCNN(input_dim, self.dropout).to(self.device)

    def _reset_history(self) -> None:
        self.history = {
            "train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []
        }

    def _overlap_fraction(self, sv_ids: Optional[np.ndarray]) -> float:
        if sv_ids is None or len(sv_ids) == 0 or len(self.training_pairs) == 0:
            return 0.0
        new_set = set(sv_ids.tolist())
        return len(new_set & self.training_pairs) / len(self.training_pairs)

    @staticmethod
    def _make_loader(
        X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
    ) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        verbose_every: int = 10,
        progress_callback: Optional[callable] = None,
        preserve_optimizer_state: bool = False,
        use_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        optimizer_name: str = "adamw",
    ) -> None:
        assert self.model is not None, "Модель не инициализирована."

        train_loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader   = self._make_loader(X_val, y_val, batch_size, shuffle=False)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer_name = optimizer_name.lower() if isinstance(optimizer_name, str) else "adamw"
        if optimizer_name == "sgd":
            optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name == "adam":
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
        else:
            optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        
        if preserve_optimizer_state and self.optimizer_state is not None:
            try:
                optimizer.load_state_dict(self.optimizer_state)
                print("[_fit] Optimizer state restored for fine-tuning")
            except Exception as e:
                print(f"[_fit] Warning: Could not restore optimizer state: {e}")
        
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=7, factor=0.5, min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        if use_early_stopping:
            print(f"[_fit] Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                preds = self.model(X_b)
                loss = criterion(preds, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(y_b)
            train_loss /= len(y_train)

            self.model.eval()
            val_loss, all_probs, all_labels = 0.0, [], []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    probs = self.model(X_b)
                    val_loss += criterion(probs, y_b).item() * len(y_b)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(y_b.cpu().numpy())
            val_loss /= len(y_val)

            all_probs  = np.array(all_probs)
            all_labels = np.array(all_labels)
            preds_bin  = (all_probs >= 0.5).astype(int)
            val_acc    = (preds_bin == all_labels).mean()
            val_auc    = _roc_auc(all_labels, all_probs)

            scheduler.step(val_loss)

            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["val_acc"].append(float(val_acc))
            self.history["val_auc"].append(float(val_auc))

            if progress_callback is not None:
                progress_callback({
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "val_auc": float(val_auc),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                })

            if epoch % verbose_every == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:>4}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Val AUC: {val_auc:.4f} | "
                    f"LR: {lr_now:.2e}"
                )
            
            if use_early_stopping:
                if val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if epoch % verbose_every == 0 or epoch == 1:
                        print(f"          → Best loss: {best_val_loss:.4f}, state saved")
                else:
                    patience_counter += 1
                    if epoch % verbose_every == 0 or epoch == 1:
                        print(f"          → No improvement ({patience_counter}/{early_stopping_patience})")
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"[_fit] Early stopping triggered at epoch {epoch}/{epochs}")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            print(f"[_fit] Restored best model weights from epoch with loss {best_val_loss:.4f}")
                        break
        
        self.optimizer_state = optimizer.state_dict()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        sv_ids: Optional[np.ndarray] = None,
        verbose_every: int = 10,
        progress_callback: Optional[callable] = None,
        use_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        optimizer_name: str = "adamw",
    ) -> None:
        print("[train] Создание модели и обучение с нуля...")
        self._build_model(X_train.shape[1])
        self._reset_history()
        if sv_ids is not None:
            self.training_pairs = set(sv_ids.tolist())
            print(f"[train] Запомнено пар sv+gene для overlap: {len(self.training_pairs):,}")
        self._fit(
            X_train, y_train, X_val, y_val,
            epochs, batch_size, lr,
            verbose_every,
            progress_callback=progress_callback,
            use_early_stopping=use_early_stopping,
            early_stopping_patience=early_stopping_patience,
            optimizer_name=optimizer_name,
        )
        print("[train] Готово.")

    def fine_tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sv_ids: Optional[np.ndarray] = None,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 5e-5,
        verbose_every: int = 5,
        progress_callback: Optional[callable] = None,
        use_early_stopping: bool = False,
        early_stopping_patience: int = 5,
        optimizer_name: str = "adamw",
    ) -> None:  
        if self.model is None:
            raise RuntimeError("Сначала вызовите train().")

        overlap = self._overlap_fraction(sv_ids)
        print(f"[fine_tune] Пересечение с обучающим набором: {overlap:.1%}")
        if overlap <= 0.5:
            print(
                "[fine_tune] ВНИМАНИЕ: overlap ≤ 50 %. "
                "Рекомендуется retrain(). Продолжаем дообучение..."
            )

        freeze_layers = [
            self.model.encoder[0],
            self.model.encoder[1],
        ]
        for layer in freeze_layers:
            for p in layer.parameters():
                p.requires_grad = False

        if sv_ids is not None:
            self.training_pairs.update(sv_ids.tolist())
            print(f"[fine_tune] Пар sv+gene в обучающем наборе: {len(self.training_pairs):,}")

        self._fit(
            X_train, y_train, X_val, y_val,
            epochs, batch_size, lr,
            verbose_every,
            progress_callback=progress_callback,
            preserve_optimizer_state=True,
            use_early_stopping=use_early_stopping,
            early_stopping_patience=early_stopping_patience,
            optimizer_name=optimizer_name,
        )

        for p in self.model.parameters():
            p.requires_grad = True

        print("[fine_tune] Дообучение завершено. Все параметры разморожены.")

    def retrain(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        sv_ids: Optional[np.ndarray] = None,
        verbose_every: int = 10,
        progress_callback: Optional[callable] = None,
        use_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        optimizer_name: str = "adamw",
    ) -> None:
        if self.model is None:
            raise RuntimeError("Сначала вызовите train().")

        overlap = self._overlap_fraction(sv_ids)
        print(f"[retrain] Пересечение с обучающим набором: {overlap:.1%}")
        print("[retrain] Сброс весов модели...")

        def _reset(module: nn.Module) -> None:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.model.apply(_reset)
        self._reset_history()

        if sv_ids is not None:
            self.training_pairs = set(sv_ids.tolist())
            print(f"[retrain] Пар sv+gene в новом наборе: {len(self.training_pairs):,}")
        else:
            self.training_pairs = set()

        self._fit(
            X_train, y_train, X_val, y_val,
            epochs, batch_size, lr,
            verbose_every,
            progress_callback=progress_callback,
            use_early_stopping=use_early_stopping,
            early_stopping_patience=early_stopping_patience,
            optimizer_name=optimizer_name,
        )
        print("[retrain] Переобучение завершено.")

    def predict(
        self,
        X: np.ndarray,
        sv_ids: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Модель не обучена. Вызовите train() сначала.")

        self.model.eval()
        batch_size = 256
        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_b = torch.from_numpy(X[i : i + batch_size]).float().to(self.device)
                probs = self.model(X_b).cpu().numpy()
                all_probs.append(probs)

        probs = np.concatenate(all_probs)
        labels = (probs >= threshold).astype(int)
        return labels, probs

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        labels, probs = self.predict(X_test, threshold=threshold)
        y_true = y_test.astype(int)

        tp = int(((labels == 1) & (y_true == 1)).sum())
        fp = int(((labels == 1) & (y_true == 0)).sum())
        fn = int(((labels == 0) & (y_true == 1)).sum())
        tn = int(((labels == 0) & (y_true == 0)).sum())

        accuracy  = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        auc = _roc_auc(y_true, probs)

        metrics = {
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "auc":       round(auc, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "model_state": self.model.state_dict() if self.model else None,
            "optimizer_state": self.optimizer_state,
            "input_dim":   self.input_dim,
            "dropout":     self.dropout,
            "sv_ids":      list(self.training_pairs),
            "history":     self.history,
        }
        torch.save(checkpoint, path)
        print(f"[save] Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str) -> "SVModel":
        checkpoint = torch.load(path, map_location="cpu")
        input_dim = checkpoint["input_dim"]
        dropout = checkpoint.get("dropout", 0.3)
        
        model = cls(input_dim=input_dim, dropout=dropout)
        
        model.model.load_state_dict(checkpoint["model_state"])
        model.training_pairs = set(checkpoint.get("sv_ids", []))
        model.history = checkpoint.get("history", model.history)
        model.optimizer_state = checkpoint.get("optimizer_state")
        model.model.eval()
        print(f"[load] Модель загружена: {path} | input_dim={input_dim}")
        
        return model


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tps = np.cumsum(y_true_sorted)
    fps = np.arange(1, len(y_true) + 1) - tps
    tpr = tps / n_pos
    fpr = fps / n_neg

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auc = float(trapezoid(tpr, fpr))
    return abs(auc)
