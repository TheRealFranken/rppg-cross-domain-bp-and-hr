"""
Deep-learning enhancer for ROI-4 rPPG.

This is a supervised pipeline:
- uses ROI 4 only
- extracts patch-level CHROM / POS / GREEN candidate signals
- trains a small temporal CNN to predict clean PPG-like waveforms

Use this when you want to test whether a learned temporal model can push
ROI-4 extraction beyond what classical fusion can do.

Typical notebook usage:

    from roi4_dl_enhancer import train_roi4_dl_model, evaluate_roi4_dl_model

    artifacts = train_roi4_dl_model(ROOT, epochs=20, top_k_patches=5)
    results = evaluate_roi4_dl_model(ROOT, artifacts)
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from roi4_rppg_advanced import (
    VIDEO_EXTS,
    bandpass,
    chrom_window,
    extract_roi4_patch_traces,
    find_subjects,
    green_rppg,
    load_ppg,
    mp_face_mesh,
    overlap_add,
    pos_window,
    prepare_pair,
    standardize,
)


def best_lag_correlation(a, b, fps, max_shift_seconds=1.5):
    a = standardize(a)
    b = standardize(b)
    max_shift = int(max_shift_seconds * fps)

    best_corr = -np.inf
    best_lag = 0

    for lag in range(-max_shift, max_shift + 1):
        if lag < 0:
            x = a[-lag:]
            y = b[: len(x)]
        elif lag > 0:
            x = a[:-lag]
            y = b[lag:]
        else:
            x = a
            y = b

        if len(x) < 10:
            continue

        corr = np.corrcoef(x, y)[0, 1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr = float(corr)
            best_lag = lag

    return best_corr, best_lag


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patch_quality(sig, fs):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < max(32, int(fs * 2)):
        return 0.0

    from scipy.signal import welch

    nperseg = min(len(sig), max(int(fs * 8), 32))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    band = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(band):
        return 0.0

    freqs = freqs[band]
    psd = psd[band]
    peak_idx = int(np.argmax(psd))
    peak_freq = freqs[peak_idx]
    signal_band = np.abs(freqs - peak_freq) <= 0.15

    signal_power = np.sum(psd[signal_band])
    noise_power = np.sum(psd[~signal_band])
    return float(signal_power / (noise_power + 1e-8))


def temporal_derivative(sig):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) == 0:
        return sig
    diff = np.diff(sig, prepend=sig[0])
    return standardize(diff)


def extract_patch_feature_bank(subject_path, face_mesh, top_k_patches=5, min_frames=200):
    videos = [name for name in os.listdir(subject_path) if name.lower().endswith(VIDEO_EXTS)]
    if not videos:
        return None

    ppg = load_ppg(subject_path)
    if ppg is None:
        return None

    video_path = os.path.join(subject_path, videos[0])
    patch_traces, fps = extract_roi4_patch_traces(video_path, face_mesh)
    if not patch_traces or len(patch_traces[0]) < min_frames:
        return None

    patch_features = []
    patch_scores = []

    for trace in patch_traces:
        chrom_sig = overlap_add(trace, fps, chrom_window)
        pos_sig = overlap_add(trace, fps, pos_window)
        green_sig = standardize(bandpass(green_rppg(trace), fps))

        q = max(
            patch_quality(chrom_sig, fps),
            patch_quality(pos_sig, fps),
            patch_quality(green_sig, fps),
        )
        patch_scores.append(q)
        patch_features.append(
            np.vstack(
                [
                    standardize(chrom_sig),
                    standardize(pos_sig),
                    standardize(green_sig),
                    temporal_derivative(chrom_sig),
                    temporal_derivative(pos_sig),
                    temporal_derivative(green_sig),
                ]
            )
        )

    top_idx = np.argsort(np.asarray(patch_scores))[::-1][:top_k_patches]
    selected = [patch_features[i] for i in top_idx]
    x = np.concatenate(selected, axis=0)
    fused_baseline = standardize(np.mean(x[: 3 * len(selected) : 3], axis=0)) if len(selected) > 0 else standardize(np.mean(x, axis=0))
    fused_baseline = standardize(np.mean(x[: min(3, x.shape[0]), :], axis=0))
    fused_derivative = temporal_derivative(fused_baseline)
    x = np.vstack([x, fused_baseline[None, :], fused_derivative[None, :]])

    y_dummy = standardize(np.mean(x, axis=0))
    _, ppg = prepare_pair(y_dummy, ppg, fps)
    if ppg is None:
        return None

    n = min(x.shape[1], len(ppg))
    x = x[:, :n]
    y = ppg[:n]

    return {
        "subject": subject_path,
        "fps": fps,
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "channels": x.shape[0],
        "length": n,
        "top_patch_indices": top_idx.tolist(),
    }


def window_quality(sig, fs):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    return patch_quality(sig, fs)


def make_windows(x, y, fs, window_size=256, stride=64, min_window_quality=0.5):
    xs = []
    ys = []
    n = len(y)
    if n < window_size:
        return xs, ys

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        xw = x[:, start:end]
        yw = y[start:end]
        if window_quality(yw, fs) < min_window_quality:
            continue
        xs.append(xw)
        ys.append(yw)

    return xs, ys


class WindowDataset(Dataset):
    def __init__(self, feature_dicts, window_size=256, stride=64, min_window_quality=0.5):
        self.samples = []
        for item in feature_dicts:
            xs, ys = make_windows(
                item["x"],
                item["y"],
                item["fps"],
                window_size=window_size,
                stride=stride,
                min_window_quality=min_window_quality,
            )
            self.samples.extend(list(zip(xs, ys)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class PulseEnhancerNet(nn.Module):
    def __init__(self, in_channels, hidden=96, depth=8):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock1D(hidden, dilation=2 ** (i % 4)) for i in range(depth)]
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x[:, 0, :]


def pearson_loss(pred, target):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    pred = pred / (pred.std(dim=-1, keepdim=True) + 1e-8)
    target = target / (target.std(dim=-1, keepdim=True) + 1e-8)

    corr = (pred * target).mean(dim=-1)
    return 1.0 - corr.mean()


def spectral_loss(pred, target):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)

    pred_mag = torch.log1p(torch.abs(pred_fft))
    target_mag = torch.log1p(torch.abs(target_fft))

    return F.l1_loss(pred_mag, target_mag)


def composite_loss(pred, target):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    pred = pred / (pred.std(dim=-1, keepdim=True) + 1e-8)
    target = target - target.mean(dim=-1, keepdim=True)
    target = target / (target.std(dim=-1, keepdim=True) + 1e-8)

    loss_corr = pearson_loss(pred, target)
    loss_l1 = F.l1_loss(pred, target)
    loss_spec = spectral_loss(pred, target)
    return 0.6 * loss_corr + 0.15 * loss_l1 + 0.25 * loss_spec


def split_subjects(items, val_ratio=0.2):
    items = list(items)
    random.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
    val_items = items[:n_val]
    train_items = items[n_val:]
    if not train_items and val_items:
        train_items = val_items[:]
        val_items = val_items[:1]
    return train_items, val_items


def build_feature_bank(
    root,
    top_k_patches=5,
    min_frames=200,
):
    feature_items = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        subjects = find_subjects(root)
        print("Subjects:", len(subjects))
        for subj in subjects:
            print("Extracting:", subj)
            item = extract_patch_feature_bank(
                subj,
                face_mesh,
                top_k_patches=top_k_patches,
                min_frames=min_frames,
            )
            if item is not None:
                feature_items.append(item)

    if not feature_items:
        raise RuntimeError("No feature items found.")
    return feature_items


def train_model_from_items(
    train_items,
    val_items,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    window_size=256,
    stride=64,
    min_window_quality=0.5,
    seed=42,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_channels = train_items[0]["channels"]
    train_ds = WindowDataset(
        train_items,
        window_size=window_size,
        stride=stride,
        min_window_quality=min_window_quality,
    )
    val_ds = WindowDataset(
        val_items,
        window_size=window_size,
        stride=stride,
        min_window_quality=min_window_quality,
    ) if val_items else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        if val_ds and len(val_ds) > 0
        else None
    )

    model = PulseEnhancerNet(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_state = None
    best_val = math.inf
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = composite_loss(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else math.nan

        val_loss = math.nan
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    loss = composite_loss(pred, yb)
                    val_losses.append(float(loss.item()))

            val_loss = float(np.mean(val_losses)) if val_losses else math.nan
            scheduler.step(val_loss)

            if np.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            if train_loss < best_val:
                best_val = train_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model.cpu(),
        "history": history,
        "in_channels": in_channels,
        "window_size": window_size,
        "stride": stride,
    }


def train_roi4_dl_model(
    root,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    window_size=256,
    stride=64,
    top_k_patches=5,
    min_frames=200,
    val_ratio=0.2,
    min_window_quality=0.5,
    seed=42,
):
    feature_items = build_feature_bank(
        root,
        top_k_patches=top_k_patches,
        min_frames=min_frames,
    )
    train_items, val_items = split_subjects(feature_items, val_ratio=val_ratio)
    artifacts = train_model_from_items(
        train_items,
        val_items,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        window_size=window_size,
        stride=stride,
        min_window_quality=min_window_quality,
        seed=seed,
    )
    for row in artifacts["history"]:
        print(f"Epoch {row['epoch']:02d} | train={row['train_loss']:.4f} | val={row['val_loss']:.4f}")

    return {
        "model": artifacts["model"],
        "history": artifacts["history"],
        "train_subjects": [item["subject"] for item in train_items],
        "val_subjects": [item["subject"] for item in val_items],
        "in_channels": artifacts["in_channels"],
        "window_size": window_size,
        "stride": stride,
        "top_k_patches": top_k_patches,
        "min_window_quality": min_window_quality,
    }


def predict_full_signal(model, x, window_size=256, stride=64, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    n = x.shape[1]
    if n < window_size:
        xb = torch.tensor(x[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(xb)[0].detach().cpu().numpy()
        return standardize(pred)

    acc = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    win = np.hanning(window_size)

    starts = list(range(0, n - window_size + 1, stride))
    if starts[-1] != n - window_size:
        starts.append(n - window_size)

    with torch.no_grad():
        for start in starts:
            end = start + window_size
            xb = torch.tensor(x[:, start:end][None, :, :], dtype=torch.float32, device=device)
            pred = model(xb)[0].detach().cpu().numpy()
            acc[start:end] += pred * win
            norm[start:end] += win

    pred = acc / (norm + 1e-8)
    return standardize(pred)


def evaluate_roi4_dl_model(root, artifacts, min_frames=200):
    model = artifacts["model"]
    top_k_patches = artifacts["top_k_patches"]
    window_size = artifacts["window_size"]
    stride = artifacts["stride"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        subjects = find_subjects(root)
        for subj in subjects:
            print("Evaluating:", subj)
            item = extract_patch_feature_bank(
                subj,
                face_mesh,
                top_k_patches=top_k_patches,
                min_frames=min_frames,
            )
            if item is None:
                continue

            pred = predict_full_signal(
                model,
                item["x"],
                window_size=window_size,
                stride=stride,
                device=device,
            )
            pred, ppg = prepare_pair(pred, item["y"], item["fps"])
            zero_corr = float(np.corrcoef(pred, ppg)[0, 1])
            best_corr, best_lag = best_lag_correlation(pred, ppg, item["fps"])

            results[subj] = {
                "rppg": pred,
                "ppg": ppg,
                "fps": item["fps"],
                "zero_corr": zero_corr,
                "best_corr": float(best_corr),
                "best_lag": int(best_lag),
                "top_patch_indices": item["top_patch_indices"],
            }

    return results


def _make_folds(items, num_folds=5, seed=42):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    folds = [[] for _ in range(num_folds)]
    for idx, item in enumerate(items):
        folds[idx % num_folds].append(item)
    return [fold for fold in folds if fold]


def _predict_ensemble(models, x, window_size, stride, device=None):
    preds = []
    for model in models:
        pred = predict_full_signal(
            model,
            x,
            window_size=window_size,
            stride=stride,
            device=device,
        )
        preds.append(standardize(pred))
    mean_pred = np.mean(np.vstack(preds), axis=0)
    return standardize(mean_pred)


def evaluate_roi4_dl_model_unbiased(
    root,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    window_size=256,
    stride=64,
    top_k_patches=5,
    min_frames=200,
    num_folds=5,
    min_window_quality=0.5,
    ensemble_seeds=(42, 52, 62),
    seed=42,
):
    feature_items = build_feature_bank(
        root,
        top_k_patches=top_k_patches,
        min_frames=min_frames,
    )
    folds = _make_folds(feature_items, num_folds=num_folds, seed=seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    fold_summaries = []

    for fold_idx, test_items in enumerate(folds, start=1):
        train_items = [item for fold in folds if fold is not test_items for item in fold]
        if not train_items:
            continue

        val_count = max(1, int(0.15 * len(train_items))) if len(train_items) > 1 else 0
        val_items = train_items[:val_count]
        train_subset = train_items[val_count:] if val_count > 0 else train_items
        if not train_subset:
            train_subset = train_items

        print(f"\nFold {fold_idx}/{len(folds)} | train={len(train_subset)} val={len(val_items)} test={len(test_items)}")

        fold_models = []
        for model_seed in ensemble_seeds:
            print(f" Training seed {model_seed}")
            artifacts = train_model_from_items(
                train_subset,
                val_items,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                window_size=window_size,
                stride=stride,
                min_window_quality=min_window_quality,
                seed=model_seed,
            )
            fold_models.append(artifacts["model"])

        fold_corrs = []
        for item in test_items:
            pred = _predict_ensemble(
                fold_models,
                item["x"],
                window_size=window_size,
                stride=stride,
                device=device,
            )
            pred, ppg = prepare_pair(pred, item["y"], item["fps"])
            zero_corr = float(np.corrcoef(pred, ppg)[0, 1])
            best_corr, best_lag = best_lag_correlation(pred, ppg, item["fps"])
            fold_corrs.append(float(best_corr))

            results[item["subject"]] = {
                "rppg": pred,
                "ppg": ppg,
                "fps": item["fps"],
                "zero_corr": zero_corr,
                "best_corr": float(best_corr),
                "best_lag": int(best_lag),
                "top_patch_indices": item["top_patch_indices"],
                "fold": fold_idx,
            }

        fold_summaries.append(
            {
                "fold": fold_idx,
                "n_test": len(test_items),
                "mean_best_corr": float(np.mean(fold_corrs)) if fold_corrs else math.nan,
                "median_best_corr": float(np.median(fold_corrs)) if fold_corrs else math.nan,
            }
        )

    overall_best = [v["best_corr"] for v in results.values()]
    overall_zero = [v["zero_corr"] for v in results.values()]
    summary = {
        "n_subjects": len(results),
        "mean_zero_corr": float(np.mean(overall_zero)) if overall_zero else math.nan,
        "mean_best_corr": float(np.mean(overall_best)) if overall_best else math.nan,
        "median_best_corr": float(np.median(overall_best)) if overall_best else math.nan,
        "fold_summaries": fold_summaries,
    }

    print("\nUnbiased summary:")
    print(summary)

    return results, summary


def plot_dl_subject_result(results, subject_key):
    data = results[subject_key]
    fps = data["fps"]
    rppg = data["rppg"]
    ppg = data["ppg"]
    t = np.arange(len(rppg)) / fps

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(t, rppg, label="DL-enhanced ROI4 rPPG")
    plt.plot(t, ppg, label="PPG", alpha=0.8)
    plt.title(subject_key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Zero-lag correlation:", data["zero_corr"])
    print("Best correlation:", data["best_corr"])
    print("Best lag (frames):", data["best_lag"])
    print("Top patch indices:", data["top_patch_indices"])
