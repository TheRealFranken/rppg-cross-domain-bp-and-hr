import os

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.signal import butter, detrend, filtfilt, resample, welch


mp_face_mesh = mp.solutions.face_mesh

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")

# Keep the ROI landmark definitions unchanged.
ROI_FOREHEAD = [151, 108, 107, 55, 8, 285, 336, 337]
ROI_LCHEEK = [126, 100, 118, 117, 116, 123, 147, 187, 205, 203]
ROI_RCHEEK = [355, 429, 358, 423, 425, 411, 376, 352, 345, 346]


def standardize(sig):
    sig = np.asarray(sig, dtype=np.float64)
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)


def bandpass(sig, fs, low=0.7, high=4.0, order=3):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)

    if len(sig) < max(9, order * 3):
        return sig.copy()

    nyq = 0.5 * float(fs)
    if nyq <= 0:
        return sig.copy()

    low = max(low, 0.05)
    high = min(high, nyq * 0.95)

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
    if len(rgb_trace) == 0:
        return rgb_trace

    out = np.zeros_like(rgb_trace, dtype=np.float64)
    for c in range(rgb_trace.shape[1]):
        out[:, c] = smooth_1d(rgb_trace[:, c], window=window)
    return out


def roi_mean_rgb(frame, pts, roi):
    h, w, _ = frame.shape
    poly = np.array([pts[i] for i in roi], dtype=np.int32)

    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    pixels = frame[mask == 255].astype(np.float64)
    if len(pixels) < 10:
        return None

    # Trim dark and highlight outliers to reduce motion/shadow noise.
    luma = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    lo, hi = np.percentile(luma, [10, 90])
    keep = (luma >= lo) & (luma <= hi)
    trimmed = pixels[keep]

    if len(trimmed) < 10:
        trimmed = pixels

    return np.median(trimmed, axis=0)


def chrom_window(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    if len(rgb) == 0:
        return np.array([], dtype=np.float64)

    rgb = detrend(rgb, axis=0, type="linear")
    rgb = rgb / (np.mean(rgb, axis=0) + 1e-8)

    x_comp = 3.0 * rgb[:, 0] - 2.0 * rgb[:, 1]
    y_comp = 1.5 * rgb[:, 0] + rgb[:, 1] - 1.5 * rgb[:, 2]

    alpha = np.std(x_comp) / (np.std(y_comp) + 1e-8)
    sig = x_comp - alpha * y_comp

    return standardize(sig)


def pos_window(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    if len(rgb) == 0:
        return np.array([], dtype=np.float64)

    rgb = detrend(rgb, axis=0, type="linear")
    normalized = rgb / (np.mean(rgb, axis=0) + 1e-8) - 1.0

    s1 = normalized[:, 1] - normalized[:, 2]
    s2 = -2.0 * normalized[:, 0] + normalized[:, 1] + normalized[:, 2]

    alpha = np.std(s1) / (np.std(s2) + 1e-8)
    sig = s1 + alpha * s2

    return standardize(sig)


def overlap_add_rppg(rgb_trace, fps, window_fn):
    rgb_trace = interpolate_rgb_trace(rgb_trace)
    n = len(rgb_trace)
    if n < 32:
        return np.zeros(n, dtype=np.float64)

    window = max(int(round(1.6 * fps)), 32)
    window = min(window, n)
    step = max(window // 2, 1)

    if n <= window:
        sig = window_fn(rgb_trace)
        return standardize(bandpass(detrend(sig, type="linear"), fps))

    win = np.hanning(window)
    acc = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    starts = list(range(0, n - window + 1, step))
    if starts[-1] != n - window:
        starts.append(n - window)

    for start in starts:
        seg = rgb_trace[start : start + window]
        sig = window_fn(seg) * win
        acc[start : start + window] += sig
        norm[start : start + window] += win

    combined = acc / (norm + 1e-8)
    combined = detrend(combined, type="linear")
    combined = bandpass(combined, fps)

    return standardize(combined)


def chrom_rppg(rgb_trace, fps):
    return overlap_add_rppg(rgb_trace, fps, chrom_window)


def signal_quality(sig, fs):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if len(sig) < max(32, int(fs * 2)):
        return 0.0

    nperseg = min(len(sig), max(int(fs * 8), 32))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    band = (freqs >= 0.7) & (freqs <= 4.0)

    if not np.any(band):
        return 0.0

    band_freqs = freqs[band]
    band_psd = psd[band]
    peak_idx = int(np.argmax(band_psd))
    peak_freq = band_freqs[peak_idx]

    signal_band = np.abs(band_freqs - peak_freq) <= 0.15
    signal_power = np.sum(band_psd[signal_band])
    noise_power = np.sum(band_psd[~signal_band])

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

    band_freqs = freqs[band]
    band_psd = psd[band]
    return float(band_freqs[int(np.argmax(band_psd))])


def extract_pulse_signal(rgb_trace, fps):
    rgb_trace = smooth_rgb_trace(interpolate_rgb_trace(rgb_trace), window=5)

    chrom_sig = chrom_rppg(rgb_trace, fps)
    pos_sig = overlap_add_rppg(rgb_trace, fps, pos_window)

    corr = np.corrcoef(chrom_sig, pos_sig)[0, 1] if len(chrom_sig) > 1 else 1.0
    if np.isfinite(corr) and corr < 0:
        pos_sig = -pos_sig

    method_scores = np.array(
        [signal_quality(chrom_sig, fps), signal_quality(pos_sig, fps)],
        dtype=np.float64,
    )

    if np.sum(method_scores) <= 0:
        method_weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        method_weights = method_scores / np.sum(method_scores)

    fused = (
        method_weights[0] * standardize(chrom_sig)
        + method_weights[1] * standardize(pos_sig)
    )
    fused = standardize(bandpass(detrend(fused, type="linear"), fps))

    return fused, method_weights, method_scores


def fuse_roi_signals(signals, fps):
    signals = [standardize(sig) for sig in signals]
    scores = np.array([signal_quality(sig, fps) for sig in signals], dtype=np.float64)
    peak_freqs = np.array([dominant_frequency(sig, fps) for sig in signals], dtype=np.float64)

    valid_freqs = peak_freqs[peak_freqs > 0]
    if len(valid_freqs) > 0:
        consensus = np.median(valid_freqs)
        consistency = 1.0 / (1.0 + 8.0 * np.abs(peak_freqs - consensus))
        scores = scores * consistency

    ref_idx = int(np.argmax(scores))
    ref = signals[ref_idx]

    aligned = []
    for sig in signals:
        corr = np.corrcoef(ref, sig)[0, 1] if len(sig) > 1 else 1.0
        aligned.append(-sig if np.isfinite(corr) and corr < 0 else sig)

    if np.sum(scores) <= 0:
        weights = np.ones(len(aligned), dtype=np.float64) / len(aligned)
    else:
        weights = scores / np.sum(scores)

    fused = np.zeros_like(aligned[0], dtype=np.float64)
    for weight, sig in zip(weights, aligned):
        fused += weight * sig

    fused = bandpass(detrend(fused, type="linear"), fps)
    fused = standardize(fused)

    return fused, weights, scores


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


def extract_rgb(video_path, face_mesh):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rgb_f = []
    rgb_l = []
    rgb_r = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(frame_rgb)

        if not res.multi_face_landmarks:
            rgb_f.append([np.nan, np.nan, np.nan])
            rgb_l.append([np.nan, np.nan, np.nan])
            rgb_r.append([np.nan, np.nan, np.nan])
            continue

        h, w, _ = frame_rgb.shape
        pts = np.array(
            [
                (int(lm.x * w), int(lm.y * h))
                for lm in res.multi_face_landmarks[0].landmark
            ],
            dtype=np.int32,
        )

        f = roi_mean_rgb(frame_rgb, pts, ROI_FOREHEAD)
        l = roi_mean_rgb(frame_rgb, pts, ROI_LCHEEK)
        r = roi_mean_rgb(frame_rgb, pts, ROI_RCHEEK)

        rgb_f.append(f if f is not None else [np.nan, np.nan, np.nan])
        rgb_l.append(l if l is not None else [np.nan, np.nan, np.nan])
        rgb_r.append(r if r is not None else [np.nan, np.nan, np.nan])

    cap.release()

    rgb_f = smooth_rgb_trace(interpolate_rgb_trace(np.asarray(rgb_f, dtype=np.float64)), window=5)
    rgb_l = smooth_rgb_trace(interpolate_rgb_trace(np.asarray(rgb_l, dtype=np.float64)), window=5)
    rgb_r = smooth_rgb_trace(interpolate_rgb_trace(np.asarray(rgb_r, dtype=np.float64)), window=5)

    return rgb_f, rgb_l, rgb_r, float(fps)


def find_subjects(root):
    subjects = []
    for dirpath, _, files in os.walk(root):
        if any(name.lower().endswith(VIDEO_EXTS) for name in files):
            subjects.append(dirpath)
    subjects.sort()
    return subjects


def process_dataset(root, min_frames=200):
    results = {}

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        subjects = find_subjects(root)
        print("Subjects:", len(subjects))

        for subj in subjects:
            videos = [
                name for name in os.listdir(subj) if name.lower().endswith(VIDEO_EXTS)
            ]
            if not videos:
                continue

            video = os.path.join(subj, videos[0])
            print("\nProcessing:", subj)

            rgb_f, rgb_l, rgb_r, fps = extract_rgb(video, face_mesh)
            if len(rgb_f) < min_frames:
                continue

            rppg_f, method_weights_f, method_scores_f = extract_pulse_signal(rgb_f, fps)
            rppg_l, method_weights_l, method_scores_l = extract_pulse_signal(rgb_l, fps)
            rppg_r, method_weights_r, method_scores_r = extract_pulse_signal(rgb_r, fps)

            fused, weights, scores = fuse_roi_signals(
                [rppg_f, rppg_l, rppg_r],
                fps,
            )

            results[subj] = {
                "rppg": fused,
                "ppg": load_ppg(subj),
                "fps": fps,
                "roi_weights": {
                    "forehead": float(weights[0]),
                    "left_cheek": float(weights[1]),
                    "right_cheek": float(weights[2]),
                },
                "roi_scores": {
                    "forehead": float(scores[0]),
                    "left_cheek": float(scores[1]),
                    "right_cheek": float(scores[2]),
                },
                "method_weights": {
                    "forehead": {
                        "chrom": float(method_weights_f[0]),
                        "pos": float(method_weights_f[1]),
                    },
                    "left_cheek": {
                        "chrom": float(method_weights_l[0]),
                        "pos": float(method_weights_l[1]),
                    },
                    "right_cheek": {
                        "chrom": float(method_weights_r[0]),
                        "pos": float(method_weights_r[1]),
                    },
                },
                "method_scores": {
                    "forehead": {
                        "chrom": float(method_scores_f[0]),
                        "pos": float(method_scores_f[1]),
                    },
                    "left_cheek": {
                        "chrom": float(method_scores_l[0]),
                        "pos": float(method_scores_l[1]),
                    },
                    "right_cheek": {
                        "chrom": float(method_scores_r[0]),
                        "pos": float(method_scores_r[1]),
                    },
                },
            }

    return results


def prepare_pair(rppg, ppg, fps):
    rppg = np.asarray(rppg, dtype=np.float64)
    ppg = np.asarray(ppg, dtype=np.float64)

    ppg = resample(ppg, len(rppg))
    rppg = standardize(bandpass(rppg, fps))
    ppg = standardize(bandpass(ppg, fps))

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


def plot_results(results):
    for subj, data in results.items():
        ppg = data["ppg"]
        if ppg is None:
            continue

        fps = data["fps"]
        rppg, ppg = prepare_pair(data["rppg"], ppg, fps)
        t = np.arange(len(rppg)) / fps

        zero_lag_corr = float(np.corrcoef(rppg, ppg)[0, 1])
        best_corr, best_lag = best_lag_correlation(rppg, ppg, fps)

        plt.figure(figsize=(12, 4))
        plt.plot(t, rppg, label="rPPG")
        plt.plot(t, ppg, label="PPG", alpha=0.8)
        plt.title(subj)
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Zero-lag correlation:", zero_lag_corr)
        print("Best correlation:", best_corr)
        print("Best lag (frames):", best_lag)
        print("ROI weights:", data["roi_weights"])
