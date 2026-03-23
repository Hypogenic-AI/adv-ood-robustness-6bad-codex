#!/usr/bin/env python3
"""End-to-end robustness study for perturbation-based neural thicket ensembles.

Pipeline:
1) Train a CIFAR-10 base model.
2) Build perturbation ensembles around base weights:
   - random Gaussian directions
   - orthogonalized random directions
   - adversarial (loss-gradient) directions
3) Evaluate clean, adversarial (PGD), OOD (SVHN) metrics.
4) Compute diversity diagnostics and bootstrap confidence intervals.
5) Save metrics/plots/report-ready artifacts in results/.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import Dataset, load_from_disk
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm


@dataclass
class Config:
    seed: int = 42
    data_root: str = "datasets"
    results_dir: str = "results"
    model_dir: str = "results/models"
    plots_dir: str = "results/plots"

    num_workers: int = 8
    pin_memory: bool = True

    train_batch_size: int = 128
    eval_batch_size: int = 256

    epochs: int = 8
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    label_smoothing: float = 0.05

    ensemble_size: int = 6
    perturb_sigma: float = 0.015

    bn_recal_batches: int = 100
    grad_direction_batches: int = 6

    pgd_eps_list: Tuple[float, ...] = (2 / 255, 4 / 255, 8 / 255)
    pgd_alpha: float = 2 / 255
    pgd_steps: int = 10

    # Keep cost manageable while still >50 samples requirement
    robust_eval_size: int = 2000

    bootstrap_samples: int = 3000
    ece_bins: int = 15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HFDatasetTorch(TorchDataset):
    def __init__(self, ds: Dataset, transform=None):
        self.ds = ds
        self.transform = transform
        if "img" in ds.column_names:
            self.image_key = "img"
        elif "image" in ds.column_names:
            self.image_key = "image"
        else:
            raise KeyError(f"Could not find image column in dataset columns: {ds.column_names}")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        img = item[self.image_key]
        if self.transform is not None:
            x = self.transform(img)
        else:
            x = torchvision.transforms.functional.pil_to_tensor(img).float() / 255.0
        y = int(item["label"])
        return x, y


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    # CIFAR-style stem for 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def tensor_flatten(params: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if mask.any():
            ece += abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.mean()
    return float(ece)


def nll_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)

    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0

    return {"auroc": float(auroc), "aupr": float(aupr), "fpr95": float(fpr95)}


def bootstrap_metric_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    metric: str = "mean",
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if metric == "mean":
            diffs[i] = x[idx].mean() - y[idx].mean()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    obs = x.mean() - y.mean()
    ci_lo, ci_hi = np.quantile(diffs, [0.025, 0.975])
    pval = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "difference": float(obs),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "p_value": float(pval),
    }


def bootstrap_auc_ci(id_scores: np.ndarray, ood_scores: np.ndarray, n_boot: int, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n_id, n_ood = len(id_scores), len(ood_scores)
    aucs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        id_idx = rng.integers(0, n_id, size=n_id)
        ood_idx = rng.integers(0, n_ood, size=n_ood)
        y_true = np.concatenate([np.ones(n_id), np.zeros(n_ood)])
        scores = np.concatenate([id_scores[id_idx], ood_scores[ood_idx]])
        aucs[i] = roc_auc_score(y_true, scores)
    return {
        "mean": float(aucs.mean()),
        "ci95_low": float(np.quantile(aucs, 0.025)),
        "ci95_high": float(np.quantile(aucs, 0.975)),
    }


def collect_logits(
    models: Sequence[nn.Module],
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for m in models:
        m.eval()

    logits_ens = []
    probs_ens = []
    pred_members = []
    labels_all = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            member_logits = []
            with autocast(enabled=amp):
                for m in models:
                    member_logits.append(m(x).float())

            stacked = torch.stack(member_logits, dim=0)  # [M, B, C]
            mean_logits = stacked.mean(dim=0)
            mean_probs = torch.softmax(mean_logits, dim=1)

            logits_ens.append(mean_logits.cpu().numpy())
            probs_ens.append(mean_probs.cpu().numpy())
            pred_members.append(stacked.argmax(dim=2).permute(1, 0).cpu().numpy())
            labels_all.append(y.cpu().numpy())

    return (
        np.concatenate(logits_ens, axis=0),
        np.concatenate(probs_ens, axis=0),
        np.concatenate(pred_members, axis=0),
        np.concatenate(labels_all, axis=0),
    )


def evaluate_clean(
    models: Sequence[nn.Module],
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    ece_bins: int,
) -> Dict[str, object]:
    logits, probs, member_preds, labels = collect_logits(models, loader, device, amp)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.int32)

    acc = float(correct.mean())
    ece = expected_calibration_error(probs, labels, ece_bins)
    nll = nll_from_probs(probs, labels)

    return {
        "accuracy": acc,
        "ece": ece,
        "nll": nll,
        "correct_per_sample": correct,
        "member_preds": member_preds,
        "labels": labels,
        "probs": probs,
    }


def pgd_attack(
    models: Sequence[nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    amp: bool,
) -> torch.Tensor:
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with autocast(enabled=amp):
            logits = 0.0
            for m in models:
                logits = logits + m(x_adv)
            logits = logits / len(models)
            loss = F.cross_entropy(logits.float(), y)

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def evaluate_pgd(
    models: Sequence[nn.Module],
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    eps: float,
    alpha: float,
    steps: int,
) -> Dict[str, object]:
    for m in models:
        m.eval()

    correct_vec = []
    with torch.no_grad():
        pass

    for x, y in tqdm(loader, desc=f"PGD eps={eps:.4f}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_adv = pgd_attack(models, x, y, eps=eps, alpha=alpha, steps=steps, amp=amp)
        with torch.no_grad(), autocast(enabled=amp):
            logits = 0.0
            for m in models:
                logits = logits + m(x_adv)
            logits = logits / len(models)
            pred = logits.argmax(dim=1)
        correct_vec.append((pred == y).cpu().numpy().astype(np.int32))

    correct_vec_np = np.concatenate(correct_vec)
    return {"robust_accuracy": float(correct_vec_np.mean()), "correct_per_sample": correct_vec_np}


def compute_diversity(member_preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # member_preds shape: [N, M]
    n, m = member_preds.shape
    if m < 2:
        return {"pairwise_disagreement": 0.0, "pairwise_error_corr": 0.0}

    dis_vals = []
    corr_vals = []

    for i in range(m):
        for j in range(i + 1, m):
            pi = member_preds[:, i]
            pj = member_preds[:, j]
            dis_vals.append((pi != pj).mean())

            ei = (pi != labels).astype(np.float64)
            ej = (pj != labels).astype(np.float64)
            if ei.std() < 1e-12 or ej.std() < 1e-12:
                corr_vals.append(0.0)
            else:
                corr_vals.append(np.corrcoef(ei, ej)[0, 1])

    return {
        "pairwise_disagreement": float(np.mean(dis_vals)),
        "pairwise_error_corr": float(np.mean(corr_vals)),
    }


def load_local_data(config: Config):
    tf_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
    )
    tf_eval = torchvision.transforms.ToTensor()

    cifar_ds = load_from_disk(str(Path(config.data_root) / "cifar10"))
    svhn_ds = load_from_disk(str(Path(config.data_root) / "svhn"))

    train_ds = HFDatasetTorch(cifar_ds["train"], transform=tf_train)
    test_ds = HFDatasetTorch(cifar_ds["test"], transform=tf_eval)
    ood_test_ds = HFDatasetTorch(svhn_ds["test"], transform=tf_eval)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    train_loader_eval = DataLoader(
        HFDatasetTorch(cifar_ds["train"], transform=tf_eval),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    ood_loader = DataLoader(
        ood_test_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return {
        "train_loader": train_loader,
        "train_loader_eval": train_loader_eval,
        "test_loader": test_loader,
        "ood_loader": ood_loader,
        "raw_cifar_train": cifar_ds["train"],
        "raw_cifar_test": cifar_ds["test"],
        "raw_svhn_test": svhn_ds["test"],
    }


def summarize_dataset(ds: Dataset, name: str) -> Dict[str, object]:
    labels = np.array(ds["label"])
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
    missing = int(np.sum(pd.isna(labels)))
    dup = int(len(labels) - len(np.unique(labels.astype(np.int64) * 100000 + np.arange(len(labels)))))
    return {
        "name": name,
        "size": len(ds),
        "missing_labels": missing,
        "class_distribution": class_dist,
        "duplicate_estimate": dup,
    }


def train_base_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, config: Config, device: torch.device):
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    amp = device.type == "cuda"
    scaler = GradScaler(enabled=amp)

    history = []
    model.to(device)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for x, y in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{config.epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += x.size(0)

        scheduler.step()
        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)

        clean_eval = evaluate_clean([model], test_loader, device, amp=amp, ece_bins=config.ece_bins)

        hist_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": clean_eval["accuracy"],
            "test_ece": clean_eval["ece"],
            "test_nll": clean_eval["nll"],
        }
        history.append(hist_row)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f} test_acc={clean_eval['accuracy']:.4f}")

    return history


def copy_model(model: nn.Module, device: torch.device) -> nn.Module:
    m = build_model()
    m.load_state_dict(model.state_dict())
    m.to(device)
    return m


def generate_random_directions(param_shapes: List[Tuple[int, ...]], k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total_dim = sum(int(np.prod(s)) for s in param_shapes)
    dirs = rng.standard_normal((k, total_dim)).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    dirs = dirs / norms
    return dirs


def orthogonalize_rows(mat: np.ndarray) -> np.ndarray:
    q_rows = []
    for i in range(mat.shape[0]):
        v = mat[i].copy()
        for q in q_rows:
            v -= np.dot(v, q) * q
        n = np.linalg.norm(v)
        if n < 1e-12:
            v = np.random.standard_normal(v.shape).astype(np.float32)
            n = np.linalg.norm(v) + 1e-12
        q_rows.append(v / n)
    return np.stack(q_rows, axis=0)


def extract_param_metadata(model: nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    shapes = [tuple(p.shape) for p in params]
    sizes = [p.numel() for p in params]
    norms = np.array([p.detach().norm().item() + 1e-12 for p in params], dtype=np.float64)
    return params, shapes, sizes, norms


def direction_to_tensors(direction: np.ndarray, shapes: List[Tuple[int, ...]], device: torch.device) -> List[torch.Tensor]:
    out = []
    idx = 0
    for s in shapes:
        n = int(np.prod(s))
        part = direction[idx : idx + n].reshape(s)
        out.append(torch.from_numpy(part).to(device=device, dtype=torch.float32))
        idx += n
    return out


def perturb_model(base: nn.Module, direction: np.ndarray, sigma: float, device: torch.device) -> nn.Module:
    m = copy_model(base, device)
    params = [p for p in m.parameters() if p.requires_grad]
    _, shapes, _, norms = extract_param_metadata(m)
    dir_tensors = direction_to_tensors(direction, shapes, device)

    with torch.no_grad():
        for p, d, nrm in zip(params, dir_tensors, norms):
            dn = d.norm().item() + 1e-12
            scaled = d * float(nrm / dn)
            p.add_(sigma * scaled)
    return m


def collect_gradient_directions(
    model: nn.Module,
    train_loader_eval: DataLoader,
    device: torch.device,
    n_dirs: int,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    shapes = [tuple(p.shape) for p in params]

    grads = []
    criterion = nn.CrossEntropyLoss()

    loader_iter = iter(train_loader_eval)
    for _ in range(n_dirs):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader_eval)
            x, y = next(loader_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        g = []
        for p in params:
            if p.grad is None:
                g.append(torch.zeros_like(p).reshape(-1))
            else:
                g.append(p.grad.detach().reshape(-1))
        gvec = torch.cat(g).cpu().numpy().astype(np.float32)
        gnorm = np.linalg.norm(gvec) + 1e-12
        grads.append(gvec / gnorm)

    return np.stack(grads, axis=0)


def recalibrate_bn(model: nn.Module, train_loader_eval: DataLoader, device: torch.device, max_batches: int = 100) -> None:
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader_eval):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            _ = model(x)
    model.eval()


def ensemble_predict_scores(models: Sequence[nn.Module], loader: DataLoader, device: torch.device, amp: bool) -> np.ndarray:
    for m in models:
        m.eval()
    scores = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Score", leave=False):
            x = x.to(device, non_blocking=True)
            with autocast(enabled=amp):
                logits = 0.0
                for m in models:
                    logits = logits + m(x)
                logits = logits / len(models)
                probs = torch.softmax(logits.float(), dim=1)
                msp = probs.max(dim=1).values
            scores.append(msp.cpu().numpy())
    return np.concatenate(scores, axis=0)


def build_ensemble(
    method: str,
    base_model: nn.Module,
    train_loader_eval: DataLoader,
    config: Config,
    device: torch.device,
    seed: int,
) -> List[nn.Module]:
    assert method in {"single", "random", "orthogonal", "adversarial"}

    if method == "single":
        return [copy_model(base_model, device)]

    params, shapes, _, _ = extract_param_metadata(base_model)
    total_dim = sum(p.numel() for p in params)

    if method == "random":
        dirs = generate_random_directions(shapes, config.ensemble_size, seed)
    elif method == "orthogonal":
        dirs = generate_random_directions(shapes, config.ensemble_size, seed)
        dirs = orthogonalize_rows(dirs)
    else:
        dirs = collect_gradient_directions(
            base_model,
            train_loader_eval=train_loader_eval,
            device=device,
            n_dirs=config.grad_direction_batches,
            seed=seed,
        )
        if dirs.shape[0] < config.ensemble_size:
            reps = int(math.ceil(config.ensemble_size / dirs.shape[0]))
            dirs = np.tile(dirs, (reps, 1))
        dirs = dirs[: config.ensemble_size]
        dirs = orthogonalize_rows(dirs)

    assert dirs.shape[1] == total_dim

    ens = []
    for i in range(config.ensemble_size):
        m = perturb_model(base_model, dirs[i], sigma=config.perturb_sigma, device=device)
        recalibrate_bn(m, train_loader_eval, device=device, max_batches=config.bn_recal_batches)
        ens.append(m)

    return ens


def run() -> None:
    start = time.time()
    config = Config()

    cwd = Path.cwd()
    assert str(cwd).endswith("adv-ood-robustness-6bad-codex"), f"Unexpected cwd: {cwd}"

    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.plots_dir).mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = get_device()
    amp = device.type == "cuda"

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        }

    env = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_info,
        "timestamp": datetime.utcnow().isoformat(),
    }
    save_json(Path(config.results_dir) / "environment.json", env)

    data = load_local_data(config)

    # Data quality and summary
    ds_summary = {
        "cifar10_train": summarize_dataset(data["raw_cifar_train"], "cifar10_train"),
        "cifar10_test": summarize_dataset(data["raw_cifar_test"], "cifar10_test"),
        "svhn_test": summarize_dataset(data["raw_svhn_test"], "svhn_test"),
    }
    save_json(Path(config.results_dir) / "dataset_summary.json", ds_summary)

    # Save representative samples metadata
    samples = {
        "cifar10_train_examples": [
            {"idx": i, "label": int(data["raw_cifar_train"][i]["label"])} for i in [0, 1, 2]
        ],
        "svhn_test_examples": [
            {"idx": i, "label": int(data["raw_svhn_test"][i]["label"])} for i in [0, 1, 2]
        ],
    }
    save_json(Path(config.results_dir) / "example_samples.json", samples)

    model = build_model().to(device)
    base_ckpt = Path(config.model_dir) / "base_model.pt"
    train_hist_path = Path(config.results_dir) / "train_history.csv"
    if base_ckpt.exists():
        model.load_state_dict(torch.load(base_ckpt, map_location=device))
        if train_hist_path.exists():
            train_hist = pd.read_csv(train_hist_path).to_dict("records")
        else:
            train_hist = []
        print("Loaded existing base model checkpoint; skipping retraining.")
    else:
        train_hist = train_base_model(
            model,
            train_loader=data["train_loader"],
            test_loader=data["test_loader"],
            config=config,
            device=device,
        )
        pd.DataFrame(train_hist).to_csv(train_hist_path, index=False)
        torch.save(model.state_dict(), base_ckpt)

    # Subset for adversarial evaluation for runtime control
    robust_loader = data["test_loader"]
    if config.robust_eval_size < len(data["raw_cifar_test"]):
        subset_idx = np.arange(config.robust_eval_size)
        subset = torch.utils.data.Subset(HFDatasetTorch(data["raw_cifar_test"], transform=torchvision.transforms.ToTensor()), subset_idx)
        robust_loader = DataLoader(
            subset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    method_specs = [
        ("single", 42),
        ("random", 101),
        ("orthogonal", 202),
        ("adversarial", 303),
    ]

    results = {}
    ood_scores = {}
    diversity_rows = []

    for method, mseed in method_specs:
        print(f"\\n=== Evaluating method: {method} ===")
        set_seed(mseed)

        models = build_ensemble(
            method=method,
            base_model=model,
            train_loader_eval=data["train_loader_eval"],
            config=config,
            device=device,
            seed=mseed,
        )

        clean = evaluate_clean(models, data["test_loader"], device=device, amp=amp, ece_bins=config.ece_bins)
        div = compute_diversity(clean["member_preds"], clean["labels"])

        # OOD via max-softmax confidence
        id_scores = clean["probs"].max(axis=1)
        ood_msp = ensemble_predict_scores(models, data["ood_loader"], device=device, amp=amp)
        ood_m = ood_metrics(id_scores, ood_msp)
        ood_ci = bootstrap_auc_ci(id_scores, ood_msp, n_boot=config.bootstrap_samples, seed=mseed)

        pgd_res = {}
        for eps in config.pgd_eps_list:
            pgd_eval = evaluate_pgd(
                models,
                robust_loader,
                device=device,
                amp=amp,
                eps=eps,
                alpha=config.pgd_alpha,
                steps=config.pgd_steps,
            )
            pgd_res[f"eps_{eps:.6f}"] = {
                "robust_accuracy": pgd_eval["robust_accuracy"],
                "correct_per_sample": pgd_eval["correct_per_sample"].tolist(),
            }

        results[method] = {
            "clean": {
                "accuracy": clean["accuracy"],
                "ece": clean["ece"],
                "nll": clean["nll"],
                "correct_per_sample": clean["correct_per_sample"].tolist(),
            },
            "diversity": div,
            "ood": {
                **ood_m,
                "auroc_bootstrap": ood_ci,
            },
            "pgd": pgd_res,
        }

        ood_scores[method] = {
            "id_scores": id_scores.tolist(),
            "ood_scores": ood_msp.tolist(),
        }

        diversity_rows.append(
            {
                "method": method,
                "pairwise_disagreement": div["pairwise_disagreement"],
                "pairwise_error_corr": div["pairwise_error_corr"],
                "ood_auroc": ood_m["auroc"],
                "pgd_eps8_acc": pgd_res[f"eps_{(8/255):.6f}"]["robust_accuracy"],
            }
        )

    # Statistical comparisons to random and single
    comparisons = {}
    base_clean = np.array(results["single"]["clean"]["correct_per_sample"])
    random_clean = np.array(results["random"]["clean"]["correct_per_sample"])
    orth_clean = np.array(results["orthogonal"]["clean"]["correct_per_sample"])
    adv_clean = np.array(results["adversarial"]["clean"]["correct_per_sample"])

    comparisons["clean_random_vs_single"] = bootstrap_metric_diff(random_clean, base_clean, config.bootstrap_samples, seed=1)
    comparisons["clean_orth_vs_random"] = bootstrap_metric_diff(orth_clean, random_clean, config.bootstrap_samples, seed=2)
    comparisons["clean_adv_vs_random"] = bootstrap_metric_diff(adv_clean, random_clean, config.bootstrap_samples, seed=3)

    key_eps = f"eps_{(8/255):.6f}"
    base_pgd = np.array(results["single"]["pgd"][key_eps]["correct_per_sample"])
    random_pgd = np.array(results["random"]["pgd"][key_eps]["correct_per_sample"])
    orth_pgd = np.array(results["orthogonal"]["pgd"][key_eps]["correct_per_sample"])
    adv_pgd = np.array(results["adversarial"]["pgd"][key_eps]["correct_per_sample"])

    comparisons["pgd8_random_vs_single"] = bootstrap_metric_diff(random_pgd, base_pgd, config.bootstrap_samples, seed=4)
    comparisons["pgd8_orth_vs_random"] = bootstrap_metric_diff(orth_pgd, random_pgd, config.bootstrap_samples, seed=5)
    comparisons["pgd8_adv_vs_random"] = bootstrap_metric_diff(adv_pgd, random_pgd, config.bootstrap_samples, seed=6)

    # Diversity-performance correlation
    div_df = pd.DataFrame(diversity_rows)
    rho_ood, p_ood = spearmanr(div_df["pairwise_disagreement"], div_df["ood_auroc"])
    rho_pgd, p_pgd = spearmanr(div_df["pairwise_disagreement"], div_df["pgd_eps8_acc"])

    analysis = {
        "comparisons": comparisons,
        "diversity_correlation": {
            "spearman_disagreement_vs_ood_auroc": {"rho": float(rho_ood), "p_value": float(p_ood)},
            "spearman_disagreement_vs_pgd8_acc": {"rho": float(rho_pgd), "p_value": float(p_pgd)},
        },
    }

    save_json(Path(config.results_dir) / "metrics.json", results)
    save_json(Path(config.results_dir) / "ood_scores.json", ood_scores)
    save_json(Path(config.results_dir) / "statistical_analysis.json", analysis)
    save_json(Path(config.results_dir) / "config.json", asdict(config))

    # Create tabular summary
    rows = []
    for method in ["single", "random", "orthogonal", "adversarial"]:
        row = {
            "method": method,
            "clean_acc": results[method]["clean"]["accuracy"],
            "ece": results[method]["clean"]["ece"],
            "nll": results[method]["clean"]["nll"],
            "ood_auroc": results[method]["ood"]["auroc"],
            "ood_aupr": results[method]["ood"]["aupr"],
            "ood_fpr95": results[method]["ood"]["fpr95"],
            "disagreement": results[method]["diversity"]["pairwise_disagreement"],
            "error_corr": results[method]["diversity"]["pairwise_error_corr"],
        }
        for eps in config.pgd_eps_list:
            row[f"pgd_acc_eps_{eps:.4f}"] = results[method]["pgd"][f"eps_{eps:.6f}"]["robust_accuracy"]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(Path(config.results_dir) / "summary_table.csv", index=False)

    # Plots
    sns.set_theme(style="whitegrid")

    # Training curves
    hist_df = pd.DataFrame(train_hist)
    plt.figure(figsize=(8, 4))
    plt.plot(hist_df["epoch"], hist_df["train_acc"], marker="o", label="Train Acc")
    plt.plot(hist_df["epoch"], hist_df["test_acc"], marker="o", label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Base Model Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(config.plots_dir) / "training_curve.png", dpi=160)
    plt.close()

    # Method comparison bar chart
    plot_df = summary_df.melt(
        id_vars=["method"],
        value_vars=["clean_acc", "ood_auroc", f"pgd_acc_eps_{(8/255):.4f}"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(9, 4))
    sns.barplot(data=plot_df, x="metric", y="value", hue="method")
    plt.ylim(0, 1)
    plt.title("Method Comparison on Key Metrics")
    plt.tight_layout()
    plt.savefig(Path(config.plots_dir) / "method_comparison_key_metrics.png", dpi=160)
    plt.close()

    # Diversity relation plots
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.regplot(data=div_df, x="pairwise_disagreement", y="ood_auroc")
    plt.title("Diversity vs OOD AUROC")
    plt.subplot(1, 2, 2)
    sns.regplot(data=div_df, x="pairwise_disagreement", y="pgd_eps8_acc")
    plt.title("Diversity vs PGD(8/255) Acc")
    plt.tight_layout()
    plt.savefig(Path(config.plots_dir) / "diversity_correlations.png", dpi=160)
    plt.close()

    runtime_sec = time.time() - start
    save_json(
        Path(config.results_dir) / "runtime.json",
        {
            "total_seconds": runtime_sec,
            "total_minutes": runtime_sec / 60.0,
        },
    )

    print("Completed research pipeline.")
    print(f"Runtime: {runtime_sec/60.0:.2f} minutes")


if __name__ == "__main__":
    run()
