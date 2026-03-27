"""
Evaluate the ROI-4 deep-learning rPPG extractor with explicit PCC, MAE, and SNR formulas.

This module uses the deep ROI-4 model from roi4_dl_enhancer.py and reports:
- raw PCC
- aligned PCC after best lag and polarity correction
- aligned MAE
- aligned SNR in dB
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from roi4_dl_enhancer import (
    best_lag_correlation,
    build_feature_bank,
    predict_full_signal,
    train_model_from_items,
)
from roi4_rppg_advanced import prepare_pair, standardize


def pcc_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return math.nan

    y_mean = np.mean(y_true)
    yhat_mean = np.mean(y_pred)
    num = np.sum((y_true - y_mean) * (y_pred - yhat_mean))
    den = np.sqrt(
        np.sum((y_true - y_mean) ** 2) * np.sum((y_pred - yhat_mean) ** 2)
    )
    if den <= 1e-12:
        return math.nan
    return float(num / den)


def mae_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return math.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def snr_db_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return math.nan

    signal_power = np.mean(y_true ** 2)
    noise_power = np.mean((y_true - y_pred) ** 2)
    if signal_power <= 1e-12 or noise_power <= 1e-12:
        return math.nan if signal_power <= 1e-12 else float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def align_by_lag(y_true, y_pred, lag):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if lag < 0:
        pred = y_pred[-lag:]
        true = y_true[: len(pred)]
    elif lag > 0:
        pred = y_pred[:-lag]
        true = y_true[lag:]
    else:
        pred = y_pred
        true = y_true

    n = min(len(true), len(pred))
    if n <= 1:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return true[:n], pred[:n]


def best_abs_lag_and_sign(y_true, y_pred, fps, max_shift_seconds=1.5):
    y_true = standardize(np.asarray(y_true, dtype=np.float64).reshape(-1))
    y_pred = standardize(np.asarray(y_pred, dtype=np.float64).reshape(-1))
    max_shift = int(max_shift_seconds * fps)

    best_abs_corr = -np.inf
    best_corr = math.nan
    best_lag = 0
    best_sign = 1.0

    for lag in range(-max_shift, max_shift + 1):
        true_seg, pred_seg = align_by_lag(y_true, y_pred, lag)
        if len(true_seg) < 10:
            continue

        corr = pcc_manual(true_seg, pred_seg)
        if not np.isfinite(corr):
            continue

        if abs(corr) > best_abs_corr:
            best_abs_corr = abs(corr)
            best_corr = corr
            best_lag = lag
            best_sign = -1.0 if corr < 0 else 1.0

    return best_lag, best_sign, best_corr


def split_subjects(items, val_ratio=0.2, seed=42):
    items = list(items)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
    val_items = items[:n_val]
    train_items = items[n_val:]
    if not train_items and val_items:
        train_items = val_items[:]
        val_items = val_items[:1]
    return train_items, val_items


def _make_folds(items, num_folds=5, seed=42):
    items = list(items)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    folds = [[] for _ in range(num_folds)]
    for idx, item in enumerate(items):
        folds[idx % num_folds].append(item)
    return [fold for fold in folds if fold]


def _predict_ensemble(models, item, window_size, stride, device=None):
    preds = []
    for model in models:
        pred = predict_full_signal(
            model,
            item["x"],
            item["baseline"],
            window_size=window_size,
            stride=stride,
            device=device,
        )
        preds.append(standardize(pred))
    return standardize(np.mean(np.vstack(preds), axis=0))


def _score_pair(pred, gt, fps):
    raw_pcc = pcc_manual(gt, pred)
    best_corr, best_lag = best_lag_correlation(pred, gt, fps)
    align_lag, align_sign, aligned_corr = best_abs_lag_and_sign(gt, pred, fps)
    gt_aligned, pred_aligned = align_by_lag(gt, align_sign * pred, align_lag)

    return {
        "pcc": raw_pcc,
        "pcc_aligned": pcc_manual(gt_aligned, pred_aligned),
        "mae_aligned": mae_manual(gt_aligned, pred_aligned),
        "snr_db_aligned": snr_db_manual(gt_aligned, pred_aligned),
        "best_corr": float(best_corr),
        "best_lag": int(best_lag),
        "align_lag": int(align_lag),
        "align_sign": int(align_sign),
        "aligned_corr": float(aligned_corr) if np.isfinite(aligned_corr) else math.nan,
        "gt_aligned": gt_aligned,
        "rppg_aligned": pred_aligned,
    }


def train_and_evaluate_dl_metrics(
    root,
    epochs=30,
    batch_size=32,
    lr=1e-3,
    window_size=320,
    stride=64,
    top_k_patches=6,
    min_frames=200,
    val_ratio=0.2,
    min_window_quality=0.6,
    seed=42,
):
    items = build_feature_bank(
        root,
        top_k_patches=top_k_patches,
        min_frames=min_frames,
    )
    train_items, val_items = split_subjects(items, val_ratio=val_ratio, seed=seed)
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

    results = {}
    for item in items:
        pred = predict_full_signal(
            artifacts["model"],
            item["x"],
            item["baseline"],
            window_size=window_size,
            stride=stride,
        )
        pred, gt = prepare_pair(pred, item["y"], item["fps"])
        if pred is None or gt is None:
            continue

        metrics = _score_pair(pred, gt, item["fps"])
        results[item["subject"]] = {
            "fps": item["fps"],
            "rppg": pred,
            "ppg": gt,
            "top_patch_indices": item["top_patch_indices"],
            **metrics,
        }

    summary = summarize_dl_metrics(results)
    return results, summary, artifacts


def evaluate_dl_metrics_unbiased(
    root,
    epochs=30,
    batch_size=32,
    lr=1e-3,
    window_size=320,
    stride=64,
    top_k_patches=6,
    min_frames=200,
    num_folds=5,
    min_window_quality=0.6,
    ensemble_seeds=(42, 52, 62),
    seed=42,
):
    items = build_feature_bank(
        root,
        top_k_patches=top_k_patches,
        min_frames=min_frames,
    )
    folds = _make_folds(items, num_folds=num_folds, seed=seed)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    results = {}
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

        for item in test_items:
            pred = _predict_ensemble(
                fold_models,
                item,
                window_size=window_size,
                stride=stride,
                device=device,
            )
            pred, gt = prepare_pair(pred, item["y"], item["fps"])
            if pred is None or gt is None:
                continue

            metrics = _score_pair(pred, gt, item["fps"])
            results[item["subject"]] = {
                "fps": item["fps"],
                "rppg": pred,
                "ppg": gt,
                "top_patch_indices": item["top_patch_indices"],
                "fold": fold_idx,
                **metrics,
            }

    summary = summarize_dl_metrics(results)
    return results, summary


def summarize_dl_metrics(results):
    pcc = [v["pcc"] for v in results.values() if np.isfinite(v["pcc"])]
    pcc_aligned = [v["pcc_aligned"] for v in results.values() if np.isfinite(v["pcc_aligned"])]
    mae_aligned = [v["mae_aligned"] for v in results.values() if np.isfinite(v["mae_aligned"])]
    snr_aligned = [v["snr_db_aligned"] for v in results.values() if np.isfinite(v["snr_db_aligned"])]
    best_corr = [v["best_corr"] for v in results.values() if np.isfinite(v["best_corr"])]

    summary = {
        "n_subjects": len(results),
        "mean_pcc": float(np.mean(pcc)) if pcc else math.nan,
        "median_pcc": float(np.median(pcc)) if pcc else math.nan,
        "mean_pcc_aligned": float(np.mean(pcc_aligned)) if pcc_aligned else math.nan,
        "median_pcc_aligned": float(np.median(pcc_aligned)) if pcc_aligned else math.nan,
        "mean_mae_aligned": float(np.mean(mae_aligned)) if mae_aligned else math.nan,
        "mean_snr_db_aligned": float(np.mean(snr_aligned)) if snr_aligned else math.nan,
        "mean_best_corr": float(np.mean(best_corr)) if best_corr else math.nan,
        "median_best_corr": float(np.median(best_corr)) if best_corr else math.nan,
    }
    print("\nDL summary:")
    print(summary)
    return summary


def plot_dl_subject_metrics(results, subject):
    item = results[subject]
    t = np.arange(len(item["rppg"])) / item["fps"]
    ta = np.arange(len(item["rppg_aligned"])) / item["fps"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(t, item["ppg"], label="GT PPG", alpha=0.8)
    axes[0].plot(t, item["rppg"], label="DL rPPG", linewidth=1.8)
    axes[0].set_title(
        f"{subject}\nRaw PCC={item['pcc']:.3f} | BestCorr={item['best_corr']:.3f} | BestLag={item['best_lag']}"
    )
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(ta, item["gt_aligned"], label="GT aligned", alpha=0.8)
    axes[1].plot(ta, item["rppg_aligned"], label="DL rPPG aligned", linewidth=1.8)
    axes[1].set_title(
        f"Aligned PCC={item['pcc_aligned']:.3f} | MAE={item['mae_aligned']:.3f} | "
        f"SNR={item['snr_db_aligned']:.2f} dB | AlignLag={item['align_lag']} | Sign={item['align_sign']}"
    )
    axes[1].grid(True)
    axes[1].legend()

    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
