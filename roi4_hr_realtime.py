"""
Real-time-friendly heart-rate training on top of the best ROI-4 pipeline.

This module keeps the ROI-4 face mesh region unchanged and builds HR
prediction on sliding windows of ROI-4 patch features. The model predicts
window-level heart rate in BPM and can also be used in a streaming
frame-by-frame setting once enough frames are buffered.

Typical notebook usage:

    from roi4_hr_realtime import evaluate_hr_model_unbiased

    results, summary = evaluate_hr_model_unbiased(
        root,
        epochs=30,
        batch_size=32,
        window_size=320,
        stride=32,
        top_k_patches=6,
        ensemble_seeds=(42, 52, 62),
    )

    summary
"""

import math
import os
import random
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import medfilt, resample, welch
from torch.utils.data import DataLoader, Dataset

from roi4_dl_enhancer import (
    normalized_rgb_features,
    patch_quality,
    set_seed,
    temporal_derivative,
    yuv_features,
)
from roi4_rppg_advanced import (
    ROI4,
    VIDEO_EXTS,
    bandpass,
    chrom_window,
    extract_patch_means,
    extract_roi4_patch_traces,
    find_subjects,
    green_rppg,
    mp_face_mesh,
    overlap_add,
    pos_window,
    standardize,
)


def load_ground_truth_bundle(folder):
    path = os.path.join(folder, "ground_truth.txt")
    if not os.path.exists(path):
        return None

    gt = np.loadtxt(path)
    gt = np.asarray(gt, dtype=np.float64)

    if gt.ndim == 1:
        return None

    if gt.shape[0] == 3:
        trace = gt[0]
        hr = gt[1]
        time = gt[2]
        source = "ground_truth.txt (rows)"
    elif gt.shape[1] == 3:
        trace = gt[:, 0]
        hr = gt[:, 1]
        time = gt[:, 2]
        source = "ground_truth.txt (cols)"
    else:
        return None

    return {
        "trace": np.asarray(trace, dtype=np.float64),
        "hr": np.asarray(hr, dtype=np.float64),
        "time": np.asarray(time, dtype=np.float64),
        "source": source,
    }


def inspect_ground_truth_format(root, limit=10):
    rows = []
    for subject in find_subjects(root)[:limit]:
        bundle = load_ground_truth_bundle(subject)
        rows.append(
            {
                "subject": subject,
                "found": bundle is not None,
                "source": None if bundle is None else bundle["source"],
                "trace_len": None if bundle is None else int(len(bundle["trace"])),
                "hr_len": None if bundle is None else int(len(bundle["hr"])),
                "time_len": None if bundle is None else int(len(bundle["time"])),
            }
        )
    return rows


def load_hr(folder):
    bundle = load_ground_truth_bundle(folder)
    if bundle is None:
        return None
    return bundle["hr"]


def resample_to_length(sig, n):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) == 0 or n <= 0:
        return np.array([], dtype=np.float64)
    if len(sig) == n:
        return sig.copy()
    return resample(sig, n)


def estimate_hr_from_signal(sig, fs, low=0.7, high=4.0):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < max(32, int(fs * 2)):
        return math.nan, 0.0, math.nan

    sig = standardize(bandpass(sig, fs, low=low, high=high))
    nperseg = min(len(sig), max(int(fs * 8), 64))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        return math.nan, 0.0, math.nan

    freqs = freqs[band]
    psd = psd[band]
    peak_idx = int(np.argmax(psd))
    peak_freq = float(freqs[peak_idx])
    bpm = 60.0 * peak_freq

    peak_mask = np.abs(freqs - peak_freq) <= 0.12
    signal_power = float(np.sum(psd[peak_mask]))
    noise_power = float(np.sum(psd[~peak_mask]))
    snr = signal_power / (noise_power + 1e-8)

    return bpm, snr, peak_freq


def extract_feature_bank_from_patch_traces(patch_traces, fps, top_k_patches=6):
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
                    *normalized_rgb_features(trace),
                    *yuv_features(trace),
                ]
            )
        )

    top_idx = np.argsort(np.asarray(patch_scores))[::-1][:top_k_patches]
    selected = [patch_features[i] for i in top_idx]
    x = np.concatenate(selected, axis=0)

    baseline_components = []
    for local_idx in range(len(selected)):
        baseline_components.append(np.mean(selected[local_idx][:3, :], axis=0))

    fused_baseline = standardize(np.mean(np.vstack(baseline_components), axis=0))
    fused_derivative = temporal_derivative(fused_baseline)
    x = np.vstack([x, fused_baseline[None, :], fused_derivative[None, :]])

    return x.astype(np.float32), fused_baseline.astype(np.float32), top_idx.tolist()


def extract_hr_feature_item(subject_path, face_mesh, top_k_patches=6, min_frames=200):
    videos = [name for name in os.listdir(subject_path) if name.lower().endswith(VIDEO_EXTS)]
    if not videos:
        return None

    hr = load_hr(subject_path)
    if hr is None:
        return None

    video_path = os.path.join(subject_path, videos[0])
    patch_traces, fps = extract_roi4_patch_traces(video_path, face_mesh)
    if not patch_traces or len(patch_traces[0]) < min_frames:
        return None

    x, baseline, top_idx = extract_feature_bank_from_patch_traces(
        patch_traces,
        fps,
        top_k_patches=top_k_patches,
    )

    n = min(x.shape[1], len(baseline))
    hr = resample_to_length(hr, n)
    valid = np.isfinite(hr)
    if not np.any(valid):
        return None
    if not np.all(valid):
        xp = np.arange(len(hr))
        hr = np.interp(xp, xp[valid], hr[valid])

    x = x[:, :n]
    baseline = baseline[:n]
    hr = hr[:n]

    return {
        "subject": subject_path,
        "fps": float(fps),
        "x": x.astype(np.float32),
        "baseline": baseline.astype(np.float32),
        "hr": hr.astype(np.float32),
        "channels": x.shape[0],
        "length": n,
        "top_patch_indices": top_idx,
    }


def build_hr_feature_bank(root, top_k_patches=6, min_frames=200):
    items = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        subjects = find_subjects(root)
        print("Subjects:", len(subjects))
        for subject in subjects:
            print("Extracting HR features:", subject)
            item = extract_hr_feature_item(
                subject,
                face_mesh,
                top_k_patches=top_k_patches,
                min_frames=min_frames,
            )
            if item is not None:
                items.append(item)

    if not items:
        raise RuntimeError("No HR feature items found.")
    return items


def make_hr_windows(item, window_size=320, stride=32, min_quality=0.6):
    x = item["x"]
    baseline = item["baseline"]
    hr = item["hr"]
    fs = item["fps"]

    samples = []
    n = len(hr)
    if n < window_size:
        return samples

    starts = list(range(0, n - window_size + 1, stride))
    if starts and starts[-1] != n - window_size:
        starts.append(n - window_size)

    for start in starts:
        end = start + window_size
        xw = x[:, start:end]
        bw = baseline[start:end]
        hrw = hr[start:end]

        classical_hr, snr, peak_freq = estimate_hr_from_signal(bw, fs)
        target_hr = float(np.nanmedian(hrw))
        quality = float(patch_quality(bw, fs))

        if not np.isfinite(target_hr):
            continue
        if not np.isfinite(classical_hr):
            continue
        if target_hr < 35.0 or target_hr > 210.0:
            continue
        if quality < min_quality:
            continue

        aux = np.asarray(
            [
                classical_hr / 200.0,
                quality / 10.0,
                snr / 10.0,
                peak_freq / 4.0 if np.isfinite(peak_freq) else 0.0,
                float(np.std(bw)),
            ],
            dtype=np.float32,
        )

        samples.append(
            {
                "x": xw.astype(np.float32),
                "aux": aux,
                "target_hr": np.float32(target_hr),
                "classical_hr": np.float32(classical_hr),
                "start": start,
                "end": end,
            }
        )

    return samples


def augment_hr_window(x, aux):
    x = x.copy()
    aux = aux.copy()

    if np.random.rand() < 0.7:
        x *= np.random.uniform(0.92, 1.08)

    if np.random.rand() < 0.7:
        x += np.random.normal(0.0, 0.01, size=x.shape)

    if np.random.rand() < 0.25:
        ch = np.random.randint(0, x.shape[0])
        x[ch] *= np.random.uniform(0.0, 0.35)

    if np.random.rand() < 0.35 and x.shape[1] > 32:
        span = np.random.randint(8, min(32, x.shape[1] // 3))
        start = np.random.randint(0, x.shape[1] - span)
        x[:, start : start + span] = 0.0

    aux += np.random.normal(0.0, 0.01, size=aux.shape)
    return x, aux


class HRWindowDataset(Dataset):
    def __init__(self, items, window_size=320, stride=32, min_quality=0.6, augment=False):
        self.samples = []
        self.augment = augment
        for item in items:
            self.samples.extend(
                make_hr_windows(
                    item,
                    window_size=window_size,
                    stride=stride,
                    min_quality=min_quality,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = sample["x"]
        aux = sample["aux"]
        if self.augment:
            x, aux = augment_hr_window(x, aux)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(aux, dtype=torch.float32),
            torch.tensor(sample["target_hr"], dtype=torch.float32),
            torch.tensor(sample["classical_hr"], dtype=torch.float32),
        )


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


class RealtimeHRNet(nn.Module):
    def __init__(self, in_channels, aux_dim=5, hidden=96, depth=6, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock1D(hidden, dilation=2 ** (i % 4)) for i in range(depth)]
        )
        self.attn = nn.Conv1d(hidden, 1, kernel_size=1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + aux_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, aux, classical_hr):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        weights = torch.softmax(self.attn(x), dim=-1)
        pooled = torch.sum(x * weights, dim=-1)
        spread = torch.std(x, dim=-1)
        features = torch.cat([pooled, spread, aux], dim=1)
        delta = 15.0 * torch.tanh(self.head(features)[:, 0])
        return classical_hr + delta


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


def hr_loss(pred_hr, target_hr):
    loss_huber = F.smooth_l1_loss(pred_hr, target_hr, beta=3.0)
    loss_l1 = F.l1_loss(pred_hr, target_hr)
    return 0.7 * loss_huber + 0.3 * loss_l1


def train_hr_model_from_items(
    train_items,
    val_items,
    epochs=30,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    window_size=320,
    stride=32,
    min_quality=0.6,
    seed=42,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_channels = train_items[0]["channels"]
    train_ds = HRWindowDataset(
        train_items,
        window_size=window_size,
        stride=stride,
        min_quality=min_quality,
        augment=True,
    )
    val_ds = HRWindowDataset(
        val_items,
        window_size=window_size,
        stride=stride,
        min_quality=min_quality,
        augment=False,
    ) if val_items else None

    if len(train_ds) == 0:
        raise RuntimeError("No HR training windows found.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    model = RealtimeHRNet(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

        for xb, auxb, yb, cb in train_loader:
            xb = xb.to(device)
            auxb = auxb.to(device)
            yb = yb.to(device)
            cb = cb.to(device)

            optimizer.zero_grad()
            pred = model(xb, auxb, cb)
            loss = hr_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else math.nan

        val_loss = math.nan
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, auxb, yb, cb in val_loader:
                    xb = xb.to(device)
                    auxb = auxb.to(device)
                    yb = yb.to(device)
                    cb = cb.to(device)
                    pred = model(xb, auxb, cb)
                    val_losses.append(float(hr_loss(pred, yb).item()))

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


def _make_folds(items, num_folds=5, seed=42):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    folds = [[] for _ in range(num_folds)]
    for idx, item in enumerate(items):
        folds[idx % num_folds].append(item)
    return [fold for fold in folds if fold]


def _predict_subject_windows(model, item, window_size, stride, min_quality, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = make_hr_windows(item, window_size=window_size, stride=stride, min_quality=min_quality)
    if not samples:
        return None

    model = model.to(device)
    model.eval()

    pred_hr = []
    gt_hr = []
    times = []
    classical_hr = []

    with torch.no_grad():
        for sample in samples:
            xb = torch.tensor(sample["x"][None, :, :], dtype=torch.float32, device=device)
            auxb = torch.tensor(sample["aux"][None, :], dtype=torch.float32, device=device)
            cb = torch.tensor([sample["classical_hr"]], dtype=torch.float32, device=device)
            pred = model(xb, auxb, cb)[0].detach().cpu().item()

            pred_hr.append(float(pred))
            gt_hr.append(float(sample["target_hr"]))
            classical_hr.append(float(sample["classical_hr"]))
            times.append(float((sample["start"] + sample["end"]) * 0.5 / item["fps"]))

    pred_hr = np.asarray(pred_hr, dtype=np.float64)
    gt_hr = np.asarray(gt_hr, dtype=np.float64)
    classical_hr = np.asarray(classical_hr, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    if len(pred_hr) >= 3:
        pred_hr = medfilt(pred_hr, kernel_size=3)

    return {
        "time_sec": times,
        "pred_hr": pred_hr,
        "gt_hr": gt_hr,
        "classical_hr": classical_hr,
    }


def _predict_subject_ensemble(models, item, window_size, stride, min_quality, device=None):
    predictions = []
    template = None
    for model in models:
        out = _predict_subject_windows(
            model,
            item,
            window_size=window_size,
            stride=stride,
            min_quality=min_quality,
            device=device,
        )
        if out is None:
            return None
        template = out
        predictions.append(out["pred_hr"])

    mean_pred = np.mean(np.vstack(predictions), axis=0)
    template["pred_hr"] = mean_pred
    return template


def _window_metrics(pred_hr, gt_hr):
    mae = float(np.mean(np.abs(pred_hr - gt_hr)))
    rmse = float(np.sqrt(np.mean((pred_hr - gt_hr) ** 2)))
    corr = float(np.corrcoef(pred_hr, gt_hr)[0, 1]) if len(pred_hr) > 1 else math.nan
    return mae, rmse, corr


def train_hr_model(
    root,
    epochs=30,
    batch_size=32,
    lr=1e-3,
    window_size=320,
    stride=32,
    top_k_patches=6,
    min_frames=200,
    val_ratio=0.2,
    min_quality=0.6,
    seed=42,
):
    items = build_hr_feature_bank(
        root,
        top_k_patches=top_k_patches,
        min_frames=min_frames,
    )
    train_items, val_items = split_subjects(items, val_ratio=val_ratio)
    artifacts = train_hr_model_from_items(
        train_items,
        val_items,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        window_size=window_size,
        stride=stride,
        min_quality=min_quality,
        seed=seed,
    )

    for row in artifacts["history"]:
        print(f"Epoch {row['epoch']:02d} | train={row['train_loss']:.4f} | val={row['val_loss']:.4f}")

    return {
        "model": artifacts["model"],
        "history": artifacts["history"],
        "window_size": window_size,
        "stride": stride,
        "top_k_patches": top_k_patches,
        "min_quality": min_quality,
        "train_subjects": [item["subject"] for item in train_items],
        "val_subjects": [item["subject"] for item in val_items],
    }


def evaluate_hr_model_unbiased(
    root,
    epochs=30,
    batch_size=32,
    lr=1e-3,
    window_size=320,
    stride=32,
    top_k_patches=6,
    min_frames=200,
    num_folds=5,
    min_quality=0.6,
    ensemble_seeds=(42, 52, 62),
    seed=42,
):
    items = build_hr_feature_bank(
        root,
        top_k_patches=top_k_patches,
        min_frames=min_frames,
    )
    folds = _make_folds(items, num_folds=num_folds, seed=seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    fold_summaries = []
    pooled_pred = []
    pooled_gt = []

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
            artifacts = train_hr_model_from_items(
                train_subset,
                val_items,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                window_size=window_size,
                stride=stride,
                min_quality=min_quality,
                seed=model_seed,
            )
            fold_models.append(artifacts["model"])

        fold_mae = []
        fold_rmse = []
        fold_corr = []

        for item in test_items:
            out = _predict_subject_ensemble(
                fold_models,
                item,
                window_size=window_size,
                stride=stride,
                min_quality=min_quality,
                device=device,
            )
            if out is None:
                continue

            pred_hr = out["pred_hr"]
            gt_hr = out["gt_hr"]
            classical_hr = out["classical_hr"]
            time_sec = out["time_sec"]

            mae, rmse, corr = _window_metrics(pred_hr, gt_hr)
            fold_mae.append(mae)
            fold_rmse.append(rmse)
            if np.isfinite(corr):
                fold_corr.append(corr)

            pooled_pred.extend(pred_hr.tolist())
            pooled_gt.extend(gt_hr.tolist())

            results[item["subject"]] = {
                "time_sec": time_sec,
                "pred_hr": pred_hr,
                "gt_hr": gt_hr,
                "classical_hr": classical_hr,
                "mae": mae,
                "rmse": rmse,
                "corr": corr,
                "fold": fold_idx,
                "top_patch_indices": item["top_patch_indices"],
            }

        fold_summaries.append(
            {
                "fold": fold_idx,
                "n_test": len(test_items),
                "mean_mae": float(np.mean(fold_mae)) if fold_mae else math.nan,
                "mean_rmse": float(np.mean(fold_rmse)) if fold_rmse else math.nan,
                "mean_corr": float(np.mean(fold_corr)) if fold_corr else math.nan,
            }
        )

    subject_mae = [value["mae"] for value in results.values()]
    subject_rmse = [value["rmse"] for value in results.values()]
    subject_corr = [value["corr"] for value in results.values() if np.isfinite(value["corr"])]
    pooled_corr = (
        float(np.corrcoef(np.asarray(pooled_pred), np.asarray(pooled_gt))[0, 1])
        if len(pooled_pred) > 1
        else math.nan
    )

    summary = {
        "n_subjects": len(results),
        "mean_subject_mae": float(np.mean(subject_mae)) if subject_mae else math.nan,
        "mean_subject_rmse": float(np.mean(subject_rmse)) if subject_rmse else math.nan,
        "mean_subject_corr": float(np.mean(subject_corr)) if subject_corr else math.nan,
        "median_subject_corr": float(np.median(subject_corr)) if subject_corr else math.nan,
        "pooled_window_corr": pooled_corr,
        "fold_summaries": fold_summaries,
    }

    print("\nUnbiased HR summary:")
    print(summary)
    return results, summary


def plot_subject_hr_result(results, subject):
    import matplotlib.pyplot as plt

    item = results[subject]
    t = item["time_sec"]
    plt.figure(figsize=(12, 4))
    plt.plot(t, item["gt_hr"], label="GT HR")
    plt.plot(t, item["classical_hr"], label="Classical ROI4 HR", alpha=0.7)
    plt.plot(t, item["pred_hr"], label="Predicted HR", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("HR (BPM)")
    plt.title(
        f"{subject}\nMAE={item['mae']:.2f} BPM | RMSE={item['rmse']:.2f} BPM | Corr={item['corr']:.3f}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


class RealtimeHRPredictor:
    def __init__(self, artifacts, fps, grid_rows=3, grid_cols=3):
        self.model = artifacts["model"]
        self.window_size = int(artifacts["window_size"])
        self.stride = int(artifacts["stride"])
        self.top_k_patches = int(artifacts["top_k_patches"])
        self.min_quality = float(artifacts["min_quality"])
        self.fps = float(fps)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.n_patches = self.grid_rows * self.grid_cols
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self.patch_buffers = [deque(maxlen=self.window_size) for _ in range(self.n_patches)]
        self.frame_count = 0
        self.last_prediction = None

    def close(self):
        self.face_mesh.close()

    def _append_nan(self):
        for idx in range(self.n_patches):
            self.patch_buffers[idx].append([np.nan, np.nan, np.nan])

    def update_from_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            self._append_nan()
        else:
            h, w, _ = frame_rgb.shape
            pts = np.array(
                [(int(lm.x * w), int(lm.y * h)) for lm in result.multi_face_landmarks[0].landmark],
                dtype=np.int32,
            )
            patch_rgbs = extract_patch_means(frame_rgb, pts, ROI4, self.grid_rows, self.grid_cols)
            if patch_rgbs is None:
                self._append_nan()
            else:
                for idx in range(self.n_patches):
                    self.patch_buffers[idx].append(patch_rgbs[idx])

        self.frame_count += 1
        if self.frame_count < self.window_size:
            return None
        if (self.frame_count - self.window_size) % self.stride != 0:
            return self.last_prediction

        patch_traces = [
            np.asarray(list(buffer), dtype=np.float64) for buffer in self.patch_buffers
        ]
        x, baseline, top_idx = extract_feature_bank_from_patch_traces(
            patch_traces,
            self.fps,
            top_k_patches=self.top_k_patches,
        )
        quality = float(patch_quality(baseline, self.fps))
        classical_hr, snr, peak_freq = estimate_hr_from_signal(baseline, self.fps)
        if not np.isfinite(classical_hr) or quality < self.min_quality:
            return self.last_prediction

        aux = np.asarray(
            [
                classical_hr / 200.0,
                quality / 10.0,
                snr / 10.0,
                peak_freq / 4.0 if np.isfinite(peak_freq) else 0.0,
                float(np.std(baseline)),
            ],
            dtype=np.float32,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(x[None, :, :], dtype=torch.float32, device=device)
            auxb = torch.tensor(aux[None, :], dtype=torch.float32, device=device)
            cb = torch.tensor([classical_hr], dtype=torch.float32, device=device)
            pred_hr = float(self.model(xb, auxb, cb)[0].detach().cpu().item())

        self.last_prediction = {
            "frame_index": self.frame_count,
            "time_sec": self.frame_count / self.fps,
            "pred_hr": pred_hr,
            "classical_hr": float(classical_hr),
            "quality": quality,
            "top_patch_indices": top_idx,
        }
        return self.last_prediction


def estimate_video_hr_realtime(video_path, artifacts, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    predictor = RealtimeHRPredictor(artifacts, fps=fps)
    outputs = []

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break

            out = predictor.update_from_frame(frame)
            if out is not None:
                outputs.append(out.copy())
    finally:
        predictor.close()
        cap.release()

    return outputs
