"""
Advanced ROI-4-only rPPG extraction.

Why this version can work better:
- keeps your chosen ROI 4 fixed
- splits ROI 4 into multiple local patches
- extracts pulse candidates per patch
- uses both CHROM and POS per patch
- scores patch quality in the pulse band
- fuses only the strongest consistent patches

Typical notebook usage:

    from roi4_rppg_advanced import process_dataset_roi4, plot_subject_result

    results = process_dataset_roi4(ROOT)
    plot_subject_result(results, list(results.keys())[0])
"""

import os

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.signal import butter, detrend, filtfilt, resample, welch


mp_face_mesh = mp.solutions.face_mesh
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")

# Fixed ROI requested by user.
ROI4 = [151, 108, 107, 55, 8, 285, 336, 337]


def standardize(sig):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)


def bandpass(sig, fs, low=0.7, high=4.0, order=3):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < max(9, order * 3):
        return sig.copy()

    nyq = 0.5 * float(fs)
    if nyq <= 0:
        return sig.copy()

    high = min(high, nyq * 0.95)
    low = max(low, 0.05)
    if low >= high:
        return sig.copy()

    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    padlen = min(len(sig) - 1, 3 * max(len(a), len(b)))
    if padlen < 1:
        return sig.copy()

    return filtfilt(b, a, sig, padlen=padlen)


def interpolate_nan_1d(sig):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    n = len(sig)
    if n == 0:
        return sig

    x = np.arange(n)
    good = np.isfinite(sig)
    if not np.any(good):
        return np.zeros(n, dtype=np.float64)
    if np.all(good):
        return sig
    return np.interp(x, x[good], sig[good])


def interpolate_rgb_trace(rgb_trace):
    rgb_trace = np.asarray(rgb_trace, dtype=np.float64)
    if rgb_trace.size == 0:
        return rgb_trace.reshape(0, 3)

    out = np.zeros_like(rgb_trace, dtype=np.float64)
    for c in range(rgb_trace.shape[1]):
        out[:, c] = interpolate_nan_1d(rgb_trace[:, c])
    return out


def smooth_1d(sig, window=5):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if window <= 1 or len(sig) < window:
        return sig.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(sig, kernel, mode="same")


def smooth_rgb_trace(rgb_trace, window=5):
    rgb_trace = np.asarray(rgb_trace, dtype=np.float64)
    out = np.zeros_like(rgb_trace, dtype=np.float64)
    for c in range(rgb_trace.shape[1]):
        out[:, c] = smooth_1d(rgb_trace[:, c], window)
    return out


def find_subjects(root):
    subjects = []
    for dirpath, _, files in os.walk(root):
        if any(name.lower().endswith(VIDEO_EXTS) for name in files):
            subjects.append(dirpath)
    subjects.sort()
    return subjects


def load_ppg(folder):
    path = os.path.join(folder, "ground_truth.txt")
    if not os.path.exists(path):
        return None

    gt = np.loadtxt(path)
    gt = np.asarray(gt)

    if gt.ndim == 1:
        return gt.astype(np.float64)
    if gt.shape[0] == 3:
        return gt[0].astype(np.float64)
    return gt[:, 0].astype(np.float64)


def chrom_window(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = detrend(rgb, axis=0, type="linear")
    rgb = rgb / (np.mean(rgb, axis=0) + 1e-8)

    x_comp = 3.0 * rgb[:, 0] - 2.0 * rgb[:, 1]
    y_comp = 1.5 * rgb[:, 0] + rgb[:, 1] - 1.5 * rgb[:, 2]
    alpha = np.std(x_comp) / (np.std(y_comp) + 1e-8)

    return standardize(x_comp - alpha * y_comp)


def pos_window(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = detrend(rgb, axis=0, type="linear")
    rgb = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1.0

    s1 = rgb[:, 1] - rgb[:, 2]
    s2 = -2.0 * rgb[:, 0] + rgb[:, 1] + rgb[:, 2]
    alpha = np.std(s1) / (np.std(s2) + 1e-8)

    return standardize(s1 + alpha * s2)


def green_rppg(rgb_trace):
    rgb_trace = np.asarray(rgb_trace, dtype=np.float64)
    green = detrend(rgb_trace[:, 1], type="linear")
    return standardize(green)


def overlap_add(rgb_trace, fps, method_fn, window_seconds=1.6):
    rgb_trace = interpolate_rgb_trace(rgb_trace)
    n = len(rgb_trace)
    if n < 32:
        return np.zeros(n, dtype=np.float64)

    window = max(int(round(window_seconds * fps)), 32)
    window = min(window, n)
    step = max(window // 2, 1)

    if n <= window:
        return standardize(bandpass(method_fn(rgb_trace), fps))

    win = np.hanning(window)
    acc = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    starts = list(range(0, n - window + 1, step))
    if starts[-1] != n - window:
        starts.append(n - window)

    for start in starts:
        seg = rgb_trace[start : start + window]
        sig = method_fn(seg) * win
        acc[start : start + window] += sig
        norm[start : start + window] += win

    out = acc / (norm + 1e-8)
    out = detrend(out, type="linear")
    out = bandpass(out, fps)
    return standardize(out)


def pulse_snr(sig, fs):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < max(32, int(fs * 2)):
        return 0.0

    nperseg = min(len(sig), max(int(fs * 8), 32))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    band = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(band):
        return 0.0

    freqs = freqs[band]
    psd = psd[band]
    peak_idx = int(np.argmax(psd))
    peak_freq = freqs[peak_idx]
    signal_band = np.abs(freqs - peak_freq) <= 0.12

    signal_power = np.sum(psd[signal_band])
    noise_power = np.sum(psd[~signal_band])
    return float(signal_power / (noise_power + 1e-8))


def dominant_frequency(sig, fs):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < max(32, int(fs * 2)):
        return 0.0

    nperseg = min(len(sig), max(int(fs * 8), 32))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    band = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(band):
        return 0.0
    return float(freqs[band][int(np.argmax(psd[band]))])


def periodicity_score(sig, fs):
    sig = standardize(sig)
    min_lag = max(int(fs / 4.0), 1)
    max_lag = max(int(fs / 0.7), min_lag + 1)

    if len(sig) <= max_lag + 2:
        return 0.0

    ac = []
    for lag in range(min_lag, max_lag):
        corr = np.corrcoef(sig[:-lag], sig[lag:])[0, 1]
        if np.isfinite(corr):
            ac.append(corr)

    if not ac:
        return 0.0
    return float(np.max(ac))


def candidate_quality(sig, fs):
    snr = pulse_snr(sig, fs)
    periodic = max(periodicity_score(sig, fs), 0.0)
    return float(snr * (1.0 + periodic))


def extract_patch_means(frame_rgb, pts, roi_idx, grid_rows=3, grid_cols=3):
    h, w, _ = frame_rgb.shape

    poly = np.array([pts[i] for i in roi_idx], dtype=np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    x, y, bw, bh = cv2.boundingRect(poly)
    if bw < grid_cols or bh < grid_rows:
        return None

    xs = np.linspace(x, x + bw, grid_cols + 1, dtype=int)
    ys = np.linspace(y, y + bh, grid_rows + 1, dtype=int)

    patch_rgbs = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            x0, x1 = xs[c], xs[c + 1]
            y0, y1 = ys[r], ys[r + 1]
            if x1 <= x0 or y1 <= y0:
                patch_rgbs.append([np.nan, np.nan, np.nan])
                continue

            patch_mask = np.zeros((h, w), dtype=np.uint8)
            patch_mask[y0:y1, x0:x1] = 255
            final_mask = cv2.bitwise_and(mask, patch_mask)

            pixels = frame_rgb[final_mask == 255].astype(np.float64)
            if len(pixels) < 12:
                patch_rgbs.append([np.nan, np.nan, np.nan])
                continue

            luma = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
            lo, hi = np.percentile(luma, [10, 90])
            keep = (luma >= lo) & (luma <= hi)
            trimmed = pixels[keep]
            if len(trimmed) < 12:
                trimmed = pixels

            patch_rgbs.append(np.median(trimmed, axis=0))

    return np.asarray(patch_rgbs, dtype=np.float64)


def extract_roi4_patch_traces(video_path, face_mesh, grid_rows=3, grid_cols=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    n_patches = grid_rows * grid_cols
    patch_traces = [[] for _ in range(n_patches)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            for i in range(n_patches):
                patch_traces[i].append([np.nan, np.nan, np.nan])
            continue

        h, w, _ = frame_rgb.shape
        pts = np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in result.multi_face_landmarks[0].landmark],
            dtype=np.int32,
        )

        patch_rgbs = extract_patch_means(frame_rgb, pts, ROI4, grid_rows=grid_rows, grid_cols=grid_cols)
        if patch_rgbs is None:
            for i in range(n_patches):
                patch_traces[i].append([np.nan, np.nan, np.nan])
            continue

        for i in range(n_patches):
            patch_traces[i].append(patch_rgbs[i])

    cap.release()

    cleaned = []
    for trace in patch_traces:
        trace = interpolate_rgb_trace(np.asarray(trace, dtype=np.float64))
        trace = smooth_rgb_trace(trace, window=5)
        cleaned.append(trace)

    return cleaned, float(fps)


def build_patch_candidate(rgb_trace, fps):
    candidates = {}

    chrom_sig = overlap_add(rgb_trace, fps, chrom_window)
    pos_sig = overlap_add(rgb_trace, fps, pos_window)
    green_sig = standardize(bandpass(green_rppg(rgb_trace), fps))

    for name, sig in [("chrom", chrom_sig), ("pos", pos_sig), ("green", green_sig)]:
        candidates[name] = {
            "signal": sig,
            "quality": candidate_quality(sig, fps),
            "peak_freq": dominant_frequency(sig, fps),
        }

    best_name = max(candidates, key=lambda k: candidates[k]["quality"])
    best = candidates[best_name]
    return best["signal"], best["quality"], best["peak_freq"], best_name, candidates


def fuse_patch_signals(patch_signals, patch_scores, patch_freqs, fps, top_k=4):
    patch_signals = [standardize(sig) for sig in patch_signals]
    patch_scores = np.asarray(patch_scores, dtype=np.float64)
    patch_freqs = np.asarray(patch_freqs, dtype=np.float64)

    valid = patch_scores > 0
    if not np.any(valid):
        fused = np.mean(np.vstack(patch_signals), axis=0)
        return standardize(bandpass(fused, fps)), np.ones(len(patch_signals)) / len(patch_signals)

    consensus = np.median(patch_freqs[valid])
    consistency = 1.0 / (1.0 + 10.0 * np.abs(patch_freqs - consensus))
    scores = patch_scores * consistency

    top_idx = np.argsort(scores)[::-1][: max(1, min(top_k, len(scores)))]
    top_scores = scores[top_idx]
    if np.sum(top_scores) <= 0:
        top_scores = np.ones(len(top_idx), dtype=np.float64)
    weights = top_scores / np.sum(top_scores)

    ref = patch_signals[top_idx[0]]
    aligned = []
    for idx in top_idx:
        sig = patch_signals[idx]
        corr = np.corrcoef(ref, sig)[0, 1] if len(sig) > 1 else 1.0
        aligned.append(-sig if np.isfinite(corr) and corr < 0 else sig)

    fused = np.zeros_like(ref, dtype=np.float64)
    for weight, sig in zip(weights, aligned):
        fused += weight * sig

    fused = standardize(bandpass(detrend(fused, type="linear"), fps))

    full_weights = np.zeros(len(patch_signals), dtype=np.float64)
    for idx, weight in zip(top_idx, weights):
        full_weights[idx] = weight

    return fused, full_weights


def prepare_pair(rppg, ppg, fps):
    ppg = np.asarray(ppg, dtype=np.float64)
    rppg = np.asarray(rppg, dtype=np.float64)
    if len(ppg) == 0 or len(rppg) == 0:
        return None, None

    ppg = resample(ppg, len(rppg))
    ppg = standardize(bandpass(ppg, fps))
    rppg = standardize(bandpass(rppg, fps))
    return rppg, ppg


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


def process_subject_roi4(subject_path, face_mesh, min_frames=200, grid_rows=3, grid_cols=3, top_k=4):
    videos = [name for name in os.listdir(subject_path) if name.lower().endswith(VIDEO_EXTS)]
    if not videos:
        return None

    video_path = os.path.join(subject_path, videos[0])
    patch_traces, fps = extract_roi4_patch_traces(
        video_path,
        face_mesh,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )

    if not patch_traces or len(patch_traces[0]) < min_frames:
        return None

    patch_signals = []
    patch_scores = []
    patch_freqs = []
    patch_methods = []
    patch_details = []

    for patch_idx, trace in enumerate(patch_traces):
        sig, score, peak_freq, best_method, candidates = build_patch_candidate(trace, fps)
        patch_signals.append(sig)
        patch_scores.append(score)
        patch_freqs.append(peak_freq)
        patch_methods.append(best_method)
        patch_details.append(
            {
                "patch": patch_idx,
                "best_method": best_method,
                "quality": float(score),
                "peak_freq": float(peak_freq),
                "candidate_qualities": {k: float(v["quality"]) for k, v in candidates.items()},
            }
        )

    fused, patch_weights = fuse_patch_signals(
        patch_signals,
        patch_scores,
        patch_freqs,
        fps,
        top_k=top_k,
    )

    ppg = load_ppg(subject_path)
    zero_corr = None
    best_corr = None
    best_lag = None
    ppg_plot = None

    if ppg is not None:
        fused_pair, ppg_pair = prepare_pair(fused, ppg, fps)
        if fused_pair is not None:
            fused = fused_pair
            ppg_plot = ppg_pair
            zero_corr = float(np.corrcoef(fused, ppg_plot)[0, 1])
            best_corr, best_lag = best_lag_correlation(fused, ppg_plot, fps)

    return {
        "subject": subject_path,
        "fps": fps,
        "rppg": fused,
        "ppg": ppg_plot,
        "zero_corr": zero_corr,
        "best_corr": best_corr,
        "best_lag": best_lag,
        "patch_weights": patch_weights,
        "patch_methods": patch_methods,
        "patch_scores": patch_scores,
        "patch_freqs": patch_freqs,
        "patch_details": patch_details,
    }


def process_dataset_roi4(root, min_frames=200, grid_rows=3, grid_cols=3, top_k=4):
    results = {}
    subjects = find_subjects(root)
    print("Subjects:", len(subjects))

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        for subj in subjects:
            print("\nProcessing:", subj)
            result = process_subject_roi4(
                subj,
                face_mesh,
                min_frames=min_frames,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                top_k=top_k,
            )
            if result is not None:
                results[subj] = result

    return results


def plot_subject_result(results, subject_key):
    data = results[subject_key]
    rppg = data["rppg"]
    ppg = data["ppg"]
    fps = data["fps"]
    t = np.arange(len(rppg)) / fps

    plt.figure(figsize=(12, 4))
    plt.plot(t, rppg, label="ROI4 rPPG")
    if ppg is not None:
        plt.plot(t, ppg, label="PPG", alpha=0.8)
    plt.title(subject_key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Zero-lag correlation:", data["zero_corr"])
    print("Best correlation:", data["best_corr"])
    print("Best lag (frames):", data["best_lag"])

    plt.figure(figsize=(10, 4))
    plt.bar([f"P{i}" for i in range(len(data["patch_weights"]))], data["patch_weights"])
    plt.title("ROI4 patch fusion weights")
    plt.ylabel("weight")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

