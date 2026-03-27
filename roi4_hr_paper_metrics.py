"""
Paper-style HR evaluation built on top of the ROI-4 deep-learning rPPG extractor.

This module keeps the best developed ROI-4 DL waveform extractor, then evaluates
heart-rate performance in the style of the referenced paper:
- MAE on window-level HR estimates (BPM)
- windowed PCC on rPPG vs reference PPG segments
- FFT-based SNR around reference HR and its first harmonic
"""

import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import medfilt, resample, welch

from roi4_dl_enhancer import (
    build_feature_bank,
    predict_full_signal,
    train_model_from_items,
)
from roi4_rppg_advanced import ROI4, VIDEO_EXTS, bandpass, mp_face_mesh, prepare_pair, standardize


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
    elif gt.shape[1] == 3:
        trace = gt[:, 0]
        hr = gt[:, 1]
        time = gt[:, 2]
    else:
        return None

    return {
        "trace": np.asarray(trace, dtype=np.float64),
        "hr": np.asarray(hr, dtype=np.float64),
        "time": np.asarray(time, dtype=np.float64),
    }


def resample_to_length(sig, n):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) == 0 or n <= 0:
        return np.array([], dtype=np.float64)
    if len(sig) == n:
        return sig.copy()
    return resample(sig, n)


def pcc_manual(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) != len(b) or len(a) < 2:
        return math.nan

    am = np.mean(a)
    bm = np.mean(b)
    num = np.sum((a - am) * (b - bm))
    den = np.sqrt(np.sum((a - am) ** 2) * np.sum((b - bm) ** 2))
    if den <= 1e-12:
        return math.nan
    return float(num / den)


def segment_bounds(n, window, step):
    if n < window:
        return []
    starts = list(range(0, n - window + 1, step))
    if starts and starts[-1] != n - window:
        starts.append(n - window)
    return [(start, start + window) for start in starts]


def estimate_hr_bpm(sig, fs, low=0.7, high=4.0):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < max(32, int(fs * 2)):
        return math.nan

    sig = standardize(bandpass(sig, fs, low=low, high=high))
    nperseg = min(len(sig), max(int(fs * 8), 64))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        return math.nan

    peak_freq = float(freqs[band][int(np.argmax(psd[band]))])
    return 60.0 * peak_freq


def _parabolic_peak(freqs, psd, idx):
    if idx <= 0 or idx >= len(psd) - 1:
        return float(freqs[idx]), float(psd[idx])

    y0, y1, y2 = psd[idx - 1], psd[idx], psd[idx + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) <= 1e-12:
        return float(freqs[idx]), float(y1)

    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    step = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    return float(freqs[idx] + delta * step), float(y1 - 0.25 * (y0 - y2) * delta)


def estimate_hr_bpm_refined(sig, fs, low=0.7, high=4.0, hr_hint_bpm=None):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < max(32, int(fs * 2)):
        return math.nan, 0.0

    sig = standardize(bandpass(sig, fs, low=low, high=high))
    nperseg = min(len(sig), max(int(fs * 8), 64))
    nfft = max(2048, int(2 ** math.ceil(math.log2(max(len(sig), nperseg)))))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, nfft=nfft)
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        return math.nan, 0.0

    freqs = freqs[band]
    psd = psd[band]
    psd = np.asarray(psd, dtype=np.float64)
    if len(psd) < 3:
        idx = int(np.argmax(psd))
        return float(60.0 * freqs[idx]), float(psd[idx])

    scores = np.zeros_like(psd)
    for idx, f in enumerate(freqs):
        harmonic = 2.0 * f
        harm_idx = int(np.argmin(np.abs(freqs - harmonic)))
        harmonic_score = 0.0
        if abs(freqs[harm_idx] - harmonic) <= 0.20:
            harmonic_score = 0.5 * psd[harm_idx]

        hint_score = 0.0
        if hr_hint_bpm is not None and np.isfinite(hr_hint_bpm):
            hint_hz = hr_hint_bpm / 60.0
            hint_score = 0.15 * psd[idx] / (1.0 + 4.0 * abs(f - hint_hz))

        scores[idx] = psd[idx] + harmonic_score + hint_score

    peak_idx = int(np.argmax(scores))
    peak_freq, peak_power = _parabolic_peak(freqs, psd, peak_idx)

    neigh = np.abs(freqs - peak_freq) <= 0.12
    sig_power = float(np.sum(psd[neigh]))
    noise_power = float(np.sum(psd[~neigh]))
    quality = sig_power / (noise_power + 1e-8)

    return float(60.0 * peak_freq), float(quality)


def paper_pcc(pred_rppg, ref_ppg, fs, window_seconds=25.0, step_seconds=None):
    pred_rppg = np.asarray(pred_rppg, dtype=np.float64).reshape(-1)
    ref_ppg = np.asarray(ref_ppg, dtype=np.float64).reshape(-1)

    if step_seconds is None:
        step_seconds = window_seconds

    window = int(round(window_seconds * fs))
    step = int(round(step_seconds * fs))
    bounds = segment_bounds(min(len(pred_rppg), len(ref_ppg)), window, step)
    if not bounds:
        return math.nan, []

    values = []
    for start, end in bounds:
        corr = pcc_manual(ref_ppg[start:end], pred_rppg[start:end])
        if np.isfinite(corr):
            values.append(corr)

    return (float(np.mean(values)) if values else math.nan), values


def paper_snr(pred_rppg, ref_hr_bpm, fs, window_seconds=25.0, step_seconds=None):
    pred_rppg = np.asarray(pred_rppg, dtype=np.float64).reshape(-1)
    ref_hr_bpm = np.asarray(ref_hr_bpm, dtype=np.float64).reshape(-1)

    if step_seconds is None:
        step_seconds = window_seconds

    window = int(round(window_seconds * fs))
    step = int(round(step_seconds * fs))
    n = min(len(pred_rppg), len(ref_hr_bpm))
    bounds = segment_bounds(n, window, step)
    if not bounds:
        return math.nan, []

    values = []
    band_half_width_hz = 12.0 / 60.0

    for start, end in bounds:
        seg = standardize(pred_rppg[start:end])
        hr_ref = float(np.nanmedian(ref_hr_bpm[start:end]))
        if not np.isfinite(hr_ref):
            continue

        hr_hz = hr_ref / 60.0
        fft_vals = np.fft.rfft(seg)
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        power = np.abs(fft_vals) ** 2

        mask_main = np.abs(freqs - hr_hz) <= band_half_width_hz
        mask_harm = np.abs(freqs - 2.0 * hr_hz) <= band_half_width_hz
        signal_mask = mask_main | mask_harm

        signal_power = float(np.sum(power[signal_mask]))
        noise_power = float(np.sum(power[~signal_mask]))
        if signal_power <= 1e-12 or noise_power <= 1e-12:
            continue

        values.append(10.0 * math.log10(signal_power / noise_power))

    return (float(np.mean(values)) if values else math.nan), values


def paper_hr_mae(pred_rppg, ref_hr_bpm, fs, window_seconds=25.0, step_seconds=None):
    pred_rppg = np.asarray(pred_rppg, dtype=np.float64).reshape(-1)
    ref_hr_bpm = np.asarray(ref_hr_bpm, dtype=np.float64).reshape(-1)

    if step_seconds is None:
        step_seconds = window_seconds

    window = int(round(window_seconds * fs))
    step = int(round(step_seconds * fs))
    n = min(len(pred_rppg), len(ref_hr_bpm))
    bounds = segment_bounds(n, window, step)
    if not bounds:
        return math.nan, [], [], []

    abs_err = []
    pred_windows = []
    ref_windows = []

    for start, end in bounds:
        pred_hr = estimate_hr_bpm(pred_rppg[start:end], fs)
        true_hr = float(np.nanmedian(ref_hr_bpm[start:end]))
        if not np.isfinite(pred_hr) or not np.isfinite(true_hr):
            continue
        abs_err.append(abs(true_hr - pred_hr))
        pred_windows.append(pred_hr)
        ref_windows.append(true_hr)

    return (
        float(np.mean(abs_err)) if abs_err else math.nan,
        pred_windows,
        ref_windows,
        abs_err,
    )


def extract_windowed_hr_series(
    pred_rppg,
    baseline_rppg,
    ref_hr_bpm,
    fs,
    window_seconds=25.0,
    step_seconds=None,
):
    pred_rppg = np.asarray(pred_rppg, dtype=np.float64).reshape(-1)
    baseline_rppg = np.asarray(baseline_rppg, dtype=np.float64).reshape(-1)
    ref_hr_bpm = np.asarray(ref_hr_bpm, dtype=np.float64).reshape(-1)

    if step_seconds is None:
        step_seconds = window_seconds

    window = int(round(window_seconds * fs))
    step = int(round(step_seconds * fs))
    n = min(len(pred_rppg), len(baseline_rppg), len(ref_hr_bpm))
    bounds = segment_bounds(n, window, step)
    if not bounds:
        return None

    pred_hr_windows = []
    ref_hr_windows = []
    abs_err = []
    qualities = []
    centers_sec = []

    prev_hr = None
    for start, end in bounds:
        pred_seg = pred_rppg[start:end]
        base_seg = baseline_rppg[start:end]
        true_hr = float(np.nanmedian(ref_hr_bpm[start:end]))
        if not np.isfinite(true_hr):
            continue

        pred_hr, pred_q = estimate_hr_bpm_refined(pred_seg, fs, hr_hint_bpm=prev_hr)
        base_hr, base_q = estimate_hr_bpm_refined(base_seg, fs, hr_hint_bpm=prev_hr)

        if not np.isfinite(pred_hr) and not np.isfinite(base_hr):
            continue
        if not np.isfinite(pred_hr):
            fused_hr = base_hr
            fused_q = base_q
        elif not np.isfinite(base_hr):
            fused_hr = pred_hr
            fused_q = pred_q
        else:
            wp = max(pred_q, 1e-6)
            wb = max(0.7 * base_q, 1e-6)
            if abs(pred_hr - base_hr) > 12.0:
                if pred_q >= 1.25 * base_q:
                    fused_hr = pred_hr
                    fused_q = pred_q
                elif base_q >= 1.25 * pred_q:
                    fused_hr = base_hr
                    fused_q = base_q
                else:
                    fused_hr = (wp * pred_hr + wb * base_hr) / (wp + wb)
                    fused_q = max(pred_q, base_q)
            else:
                fused_hr = (wp * pred_hr + wb * base_hr) / (wp + wb)
                fused_q = max(pred_q, base_q)

        if prev_hr is not None and np.isfinite(prev_hr):
            fused_hr = 0.75 * fused_hr + 0.25 * prev_hr
        prev_hr = fused_hr

        pred_hr_windows.append(float(fused_hr))
        ref_hr_windows.append(float(true_hr))
        abs_err.append(abs(float(true_hr) - float(fused_hr)))
        qualities.append(float(fused_q))
        centers_sec.append(float((start + end) * 0.5 / fs))

    if len(pred_hr_windows) >= 3:
        kernel = 5 if len(pred_hr_windows) >= 5 else 3
        pred_hr_windows = medfilt(np.asarray(pred_hr_windows, dtype=np.float64), kernel_size=kernel)
        pred_hr_windows = np.asarray(pred_hr_windows, dtype=np.float64)
        ref_hr_windows = np.asarray(ref_hr_windows, dtype=np.float64)
        abs_err = np.abs(ref_hr_windows - pred_hr_windows)
    else:
        pred_hr_windows = np.asarray(pred_hr_windows, dtype=np.float64)
        ref_hr_windows = np.asarray(ref_hr_windows, dtype=np.float64)
        abs_err = np.asarray(abs_err, dtype=np.float64)

    return {
        "pred_hr_windows": pred_hr_windows,
        "ref_hr_windows": ref_hr_windows,
        "abs_err": abs_err,
        "quality_windows": np.asarray(qualities, dtype=np.float64),
        "window_centers_sec": np.asarray(centers_sec, dtype=np.float64),
        "paper_mae_bpm": float(np.mean(abs_err)) if len(abs_err) else math.nan,
    }


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


def _score_subject(pred_rppg, baseline_rppg, ref_ppg, ref_hr, fs, window_seconds=25.0, step_seconds=None):
    ref_hr = resample_to_length(ref_hr, len(pred_rppg))
    if len(ref_hr) == 0:
        return None

    valid = np.isfinite(ref_hr)
    if not np.any(valid):
        return None
    if not np.all(valid):
        xp = np.arange(len(ref_hr))
        ref_hr = np.interp(xp, xp[valid], ref_hr[valid])

    pcc, pcc_segments = paper_pcc(pred_rppg, ref_ppg, fs, window_seconds, step_seconds)
    snr_db, snr_segments = paper_snr(pred_rppg, ref_hr, fs, window_seconds, step_seconds)
    hr_series = extract_windowed_hr_series(
        pred_rppg,
        baseline_rppg,
        ref_hr,
        fs,
        window_seconds,
        step_seconds,
    )
    if hr_series is None:
        return None

    return {
        "paper_pcc": pcc,
        "paper_snr_db": snr_db,
        "paper_mae_bpm": hr_series["paper_mae_bpm"],
        "pred_hr_windows": hr_series["pred_hr_windows"],
        "ref_hr_windows": hr_series["ref_hr_windows"],
        "pcc_segments": pcc_segments,
        "snr_segments": snr_segments,
        "abs_err_segments": hr_series["abs_err"],
        "quality_windows": hr_series["quality_windows"],
        "window_centers_sec": hr_series["window_centers_sec"],
    }


def summarize_hr_paper_metrics(results):
    mae = [v["paper_mae_bpm"] for v in results.values() if np.isfinite(v["paper_mae_bpm"])]
    pcc = [v["paper_pcc"] for v in results.values() if np.isfinite(v["paper_pcc"])]
    snr = [v["paper_snr_db"] for v in results.values() if np.isfinite(v["paper_snr_db"])]

    summary = {
        "n_subjects": len(results),
        "mean_paper_mae_bpm": float(np.mean(mae)) if mae else math.nan,
        "median_paper_mae_bpm": float(np.median(mae)) if mae else math.nan,
        "mean_paper_pcc": float(np.mean(pcc)) if pcc else math.nan,
        "median_paper_pcc": float(np.median(pcc)) if pcc else math.nan,
        "mean_paper_snr_db": float(np.mean(snr)) if snr else math.nan,
        "median_paper_snr_db": float(np.median(snr)) if snr else math.nan,
    }
    print("\nPaper-style HR summary:")
    print(summary)
    return summary


def evaluate_hr_paper_metrics_unbiased(
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
    window_seconds=25.0,
    step_seconds=None,
    seed=42,
):
    items = build_feature_bank(
        root,
        top_k_patches=top_k_patches,
        min_frames=min_frames,
    )
    folds = _make_folds(items, num_folds=num_folds, seed=seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
            pred, ref_ppg = prepare_pair(pred, item["y"], item["fps"])
            if pred is None or ref_ppg is None:
                continue
            baseline_rppg, _ = prepare_pair(item["baseline"], item["y"], item["fps"])
            if baseline_rppg is None:
                continue

            bundle = load_ground_truth_bundle(item["subject"])
            if bundle is None:
                continue

            scored = _score_subject(
                pred,
                baseline_rppg,
                ref_ppg,
                bundle["hr"],
                item["fps"],
                window_seconds=window_seconds,
                step_seconds=step_seconds,
            )
            if scored is None:
                continue

            results[item["subject"]] = {
                "fps": item["fps"],
                "rppg": pred,
                "ppg": ref_ppg,
                "fold": fold_idx,
                "top_patch_indices": item["top_patch_indices"],
                **scored,
            }

    summary = summarize_hr_paper_metrics(results)
    return results, summary


def plot_hr_paper_subject(results, subject):
    item = results[subject]
    if len(item["pred_hr_windows"]) == 0:
        print("No HR windows available for this subject.")
        return

    x = np.arange(len(item["pred_hr_windows"]))
    plt.figure(figsize=(12, 4))
    plt.plot(x, item["ref_hr_windows"], label="Reference HR")
    plt.plot(x, item["pred_hr_windows"], label="Predicted HR", linewidth=2)
    plt.title(
        f"{subject}\nMAE={item['paper_mae_bpm']:.2f} BPM | PCC={item['paper_pcc']:.3f} | "
        f"SNR={item['paper_snr_db']:.2f} dB"
    )
    plt.xlabel("Window Index")
    plt.ylabel("Heart Rate (BPM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _subject_video_path(subject_path):
    videos = [name for name in os.listdir(subject_path) if name.lower().endswith(VIDEO_EXTS)]
    if not videos:
        return None
    videos.sort()
    return os.path.join(subject_path, videos[0])


def show_random_frame_with_hr(results, subject=None, seed=None):
    rng = np.random.default_rng(seed)
    if not results:
        raise ValueError("Results are empty.")

    if subject is None:
        subject = list(results.keys())[int(rng.integers(0, len(results)))]
    if subject not in results:
        raise KeyError(f"Subject not found in results: {subject}")

    item = results[subject]
    if len(item.get("window_centers_sec", [])) == 0:
        raise ValueError("No HR windows stored for this subject. Re-run evaluation with the updated module.")

    window_idx = int(rng.integers(0, len(item["window_centers_sec"])))
    center_sec = float(item["window_centers_sec"][window_idx])
    pred_hr = float(item["pred_hr_windows"][window_idx])
    ref_hr = float(item["ref_hr_windows"][window_idx])

    video_path = _subject_video_path(subject)
    if video_path is None:
        raise FileNotFoundError(f"No video found for subject: {subject}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or item["fps"]
    frame_idx = max(0, int(round(center_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    overlay = frame_rgb.copy()
    polygon = None

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        result = face_mesh.process(frame_rgb)
        if result.multi_face_landmarks:
            h, w, _ = frame_rgb.shape
            pts = np.array(
                [(int(lm.x * w), int(lm.y * h)) for lm in result.multi_face_landmarks[0].landmark],
                dtype=np.int32,
            )
            polygon = np.array([pts[i] for i in ROI4], dtype=np.int32)
            cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            fill = overlay.copy()
            cv2.fillPoly(fill, [polygon], color=(0, 255, 0))
            overlay = cv2.addWeighted(fill, 0.18, overlay, 0.82, 0.0)

    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(
        f"{os.path.basename(subject)} | frame {frame_idx} | t={center_sec:.2f}s\n"
        f"Our HR={pred_hr:.2f} BPM | Dataset HR={ref_hr:.2f} BPM | "
        f"AbsErr={abs(pred_hr - ref_hr):.2f} BPM"
    )
    plt.tight_layout()
    plt.show()

    if polygon is None:
        print("Face/ROI was not detected on this frame, but the frame was shown.")

    return {
        "subject": subject,
        "frame_index": frame_idx,
        "time_sec": center_sec,
        "pred_hr_bpm": pred_hr,
        "dataset_hr_bpm": ref_hr,
        "abs_err_bpm": abs(pred_hr - ref_hr),
        "video_path": video_path,
    }


def paper_comparison_report(summary):
    lines = []
    lines.append("Paper comparison report")
    lines.append("-----------------------")
    lines.append("Paper recommendation:")
    lines.append("- Best ROI: glabella (ROI 4)")
    lines.append("- Strong ROI group: glabella, medial forehead, lateral forehead, malars, upper nasal dorsum")
    lines.append("- Acceptable HR criterion in the paper: MAE <= 10 BPM")
    lines.append("- Reported glabella acceptance rate: 49.2% on motion, 70.5% on cognitive")
    lines.append("")
    lines.append("Our setup:")
    lines.append("- ROI used: ROI 4 / glabella")
    lines.append("- Extractor: ROI4 deep-learning rPPG pipeline")
    lines.append("- Evaluation style: paper-style HR MAE, windowed PCC, FFT-based SNR")
    lines.append("")
    lines.append("Our results:")
    lines.append(f"- Mean HR MAE: {summary.get('mean_paper_mae_bpm', math.nan):.3f} BPM")
    lines.append(f"- Median HR MAE: {summary.get('median_paper_mae_bpm', math.nan):.3f} BPM")
    lines.append(f"- Mean windowed PCC: {summary.get('mean_paper_pcc', math.nan):.3f}")
    lines.append(f"- Median windowed PCC: {summary.get('median_paper_pcc', math.nan):.3f}")
    lines.append(f"- Mean SNR: {summary.get('mean_paper_snr_db', math.nan):.3f} dB")
    lines.append(f"- Median SNR: {summary.get('median_paper_snr_db', math.nan):.3f} dB")
    lines.append("")
    mean_mae = summary.get("mean_paper_mae_bpm", math.nan)
    if np.isfinite(mean_mae):
        if mean_mae <= 10.0:
            lines.append("Interpretation:")
            lines.append("- The mean HR error is within the paper's acceptable HR threshold.")
        else:
            lines.append("Interpretation:")
            lines.append("- The mean HR error is still above the paper's acceptable HR threshold.")
    else:
        lines.append("Interpretation:")
        lines.append("- HR MAE is not available.")

    mean_pcc = summary.get("mean_paper_pcc", math.nan)
    if np.isfinite(mean_pcc):
        if mean_pcc >= 0.75:
            lines.append("- The waveform-to-reference correlation is strong for ROI4-only HR estimation.")
        elif mean_pcc >= 0.60:
            lines.append("- The waveform-to-reference correlation is moderate and usable, but can still improve.")
        else:
            lines.append("- The waveform-to-reference correlation is still weak and likely limiting HR extraction.")

    mean_snr = summary.get("mean_paper_snr_db", math.nan)
    if np.isfinite(mean_snr):
        if mean_snr >= 5.0:
            lines.append("- The signal quality is in a reasonable range for downstream HR estimation.")
        elif mean_snr >= 3.0:
            lines.append("- The signal quality is usable but still somewhat noise-limited.")
        else:
            lines.append("- The signal quality is still low and likely hurting HR robustness.")

    report = "\n".join(lines)
    print(report)
    return report
