"""
ROI-4 rPPG evaluation with explicit PCC, MAE, and SNR formulas.

This module keeps ROI 4 fixed and compares several post-processing options:
- raw fused ROI-4 rPPG
- Savitzky-Golay smoothing
- moving-average smoothing
- FFT-band filtering

Metrics are computed explicitly from formulas:
- PCC
- MAE
- SNR in dB
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from roi4_rppg_advanced import (
    VIDEO_EXTS,
    best_lag_correlation,
    bandpass,
    find_subjects,
    load_ppg,
    mp_face_mesh,
    prepare_pair,
    process_subject_roi4,
    standardize,
)


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


def moving_average(sig, window=9):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if window <= 1 or len(sig) < window:
        return sig.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(sig, kernel, mode="same")


def savgol_smooth(sig, window_length=11, polyorder=2):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < 5:
        return sig.copy()

    window_length = max(5, int(window_length))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(sig) if len(sig) % 2 == 1 else len(sig) - 1)
    if window_length <= polyorder:
        return sig.copy()

    return savgol_filter(sig, window_length=window_length, polyorder=polyorder, mode="interp")


def fft_band_filter(sig, fs, low=0.7, high=4.0):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    n = len(sig)
    if n < 4 or fs <= 0:
        return sig.copy()

    centered = sig - np.mean(sig)
    spec = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    keep = (freqs >= low) & (freqs <= high)
    spec[~keep] = 0.0
    out = np.fft.irfft(spec, n=n)
    return np.asarray(out, dtype=np.float64)


def postprocess_signal(sig, fs, method):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)

    if method == "raw":
        out = sig.copy()
    elif method == "savgol":
        out = savgol_smooth(sig, window_length=max(7, int(round(0.35 * fs)) | 1), polyorder=2)
    elif method == "moving_average":
        out = moving_average(sig, window=max(5, int(round(0.30 * fs)) | 1))
    elif method == "fft":
        out = fft_band_filter(sig, fs, low=0.7, high=4.0)
    elif method == "savgol_fft":
        out = fft_band_filter(savgol_smooth(sig, window_length=max(7, int(round(0.35 * fs)) | 1), polyorder=2), fs)
    elif method == "ma_fft":
        out = fft_band_filter(moving_average(sig, window=max(5, int(round(0.30 * fs)) | 1)), fs)
    else:
        raise ValueError(f"Unknown method: {method}")

    out = standardize(bandpass(out, fs))
    return out


def evaluate_subject_methods(subject_path, face_mesh, min_frames=200, methods=None):
    if methods is None:
        methods = ["raw", "savgol", "moving_average", "fft", "savgol_fft", "ma_fft"]

    result = process_subject_roi4(
        subject_path,
        face_mesh,
        min_frames=min_frames,
        grid_rows=3,
        grid_cols=3,
        top_k=4,
    )
    if result is None:
        return None

    base_rppg = result["rppg"]
    ppg = load_ppg(subject_path)
    if ppg is None:
        return None

    subject_metrics = {}
    for method in methods:
        pred = postprocess_signal(base_rppg, result["fps"], method)
        pred, gt = prepare_pair(pred, ppg, result["fps"])
        if pred is None or gt is None:
            continue

        pcc = pcc_manual(gt, pred)
        best_corr, best_lag = best_lag_correlation(pred, gt, result["fps"])
        align_lag, align_sign, aligned_corr = best_abs_lag_and_sign(gt, pred, result["fps"])
        gt_aligned, pred_aligned = align_by_lag(gt, align_sign * pred, align_lag)

        pcc_aligned = pcc_manual(gt_aligned, pred_aligned)
        mae_aligned = mae_manual(gt_aligned, pred_aligned)
        snr_db_aligned = snr_db_manual(gt_aligned, pred_aligned)

        subject_metrics[method] = {
            "rppg": pred,
            "ppg": gt,
            "pcc": pcc,
            "pcc_aligned": pcc_aligned,
            "mae_aligned": mae_aligned,
            "snr_db_aligned": snr_db_aligned,
            "best_corr": float(best_corr),
            "best_lag": int(best_lag),
            "align_lag": int(align_lag),
            "align_sign": int(align_sign),
            "aligned_corr": float(aligned_corr) if np.isfinite(aligned_corr) else math.nan,
            "gt_aligned": gt_aligned,
            "rppg_aligned": pred_aligned,
        }

    if not subject_metrics:
        return None

    return {
        "subject": subject_path,
        "fps": result["fps"],
        "methods": subject_metrics,
    }


def run_roi4_rppg_metrics(root, min_frames=200, methods=None):
    if methods is None:
        methods = ["raw", "savgol", "moving_average", "fft", "savgol_fft", "ma_fft"]

    results = {}
    summary_rows = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        subjects = find_subjects(root)
        print("Subjects:", len(subjects))
        for subject in subjects:
            print("Evaluating:", subject)
            item = evaluate_subject_methods(
                subject,
                face_mesh,
                min_frames=min_frames,
                methods=methods,
            )
            if item is None:
                continue
            results[subject] = item

    for method in methods:
        pccs = []
        pccs_aligned = []
        maes = []
        snrs = []
        best_corrs = []

        for item in results.values():
            if method not in item["methods"]:
                continue
            m = item["methods"][method]
            if np.isfinite(m["pcc"]):
                pccs.append(m["pcc"])
            if np.isfinite(m["pcc_aligned"]):
                pccs_aligned.append(m["pcc_aligned"])
            if np.isfinite(m["mae_aligned"]):
                maes.append(m["mae_aligned"])
            if np.isfinite(m["snr_db_aligned"]):
                snrs.append(m["snr_db_aligned"])
            if np.isfinite(m["best_corr"]):
                best_corrs.append(m["best_corr"])

        summary_rows.append(
            {
                "method": method,
                "n_subjects": len(pccs),
                "mean_pcc": float(np.mean(pccs)) if pccs else math.nan,
                "median_pcc": float(np.median(pccs)) if pccs else math.nan,
                "mean_pcc_aligned": float(np.mean(pccs_aligned)) if pccs_aligned else math.nan,
                "median_pcc_aligned": float(np.median(pccs_aligned)) if pccs_aligned else math.nan,
                "mean_mae": float(np.mean(maes)) if maes else math.nan,
                "mean_snr_db": float(np.mean(snrs)) if snrs else math.nan,
                "mean_best_corr": float(np.mean(best_corrs)) if best_corrs else math.nan,
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: row["mean_pcc_aligned"], reverse=True)
    print("\nSummary:")
    for row in summary_rows:
        print(row)

    return results, summary_rows


def plot_subject_method_comparison(results, subject, methods=None):
    item = results[subject]
    if methods is None:
        methods = list(item["methods"].keys())

    n = len(methods)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        data = item["methods"][method]
        t = np.arange(len(data["rppg"])) / item["fps"]
        ax.plot(t, data["ppg"], label="GT PPG", alpha=0.8)
        ax.plot(t, data["rppg"], label=f"rPPG ({method})", linewidth=1.8)
        ax.set_title(
            f"{os.path.basename(subject)} | {method} | "
            f"PCC={data['pcc']:.3f} | PCC_aligned={data['pcc_aligned']:.3f} | "
            f"MAE={data['mae_aligned']:.3f} | SNR={data['snr_db_aligned']:.2f} dB"
        )
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_method_summary(summary_rows):
    methods = [row["method"] for row in summary_rows]
    pcc = [row["mean_pcc_aligned"] for row in summary_rows]
    mae = [row["mean_mae"] for row in summary_rows]
    snr = [row["mean_snr_db"] for row in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].bar(methods, pcc)
    axes[0].set_title("Mean PCC (Aligned)")
    axes[0].grid(True, axis="y")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(methods, mae)
    axes[1].set_title("Mean MAE")
    axes[1].grid(True, axis="y")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(methods, snr)
    axes[2].set_title("Mean SNR (dB)")
    axes[2].grid(True, axis="y")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()
