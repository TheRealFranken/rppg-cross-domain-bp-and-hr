"""
Benchmark 28 facial ROIs for rPPG quality against ground-truth PPG.

Typical Colab usage:

    from roi_benchmark_28 import run_roi_benchmark

    ROOT = "/content/UBFC_DATASET"
    subject_df, summary_df = run_roi_benchmark(ROOT)

This script:
1. extracts RGB traces for all 28 ROIs in a single face-mesh pass
2. generates an rPPG signal per ROI
3. compares each ROI signal to ground-truth PPG
4. plots ROI ranking by correlation
5. saves CSV summaries for later analysis
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rppg_improved import (
    VIDEO_EXTS,
    best_lag_correlation,
    bandpass,
    extract_pulse_signal,
    interpolate_rgb_trace,
    load_ppg,
    mp_face_mesh,
    roi_mean_rgb,
    smooth_rgb_trace,
    standardize,
)


ROI_MAP = {
    1: [10, 109, 108, 151, 337, 338],
    2: [67, 103, 104, 105, 66, 107, 108, 109],
    3: [297, 338, 337, 336, 296, 334, 333, 332],
    4: [151, 108, 107, 55, 8, 285, 336, 337],
    5: [8, 55, 193, 122, 196, 197, 419, 351, 417, 285],
    6: [197, 196, 3, 51, 5, 281, 248, 419],
    7: [4, 45, 134, 220, 237, 44, 1, 274, 457, 440, 363, 275],
    8: [134, 131, 49, 102, 64, 219, 218, 237, 220],
    9: [363, 440, 457, 438, 439, 294, 331, 279, 360],
    10: [5, 51, 45, 4, 275, 281],
    11: [3, 217, 126, 209, 131, 134],
    12: [248, 363, 360, 429, 355, 437],
    13: [188, 114, 217, 236, 196],
    14: [412, 419, 456, 437, 343],
    15: [2, 97, 167, 37, 0, 267, 393, 326],
    16: [197, 165, 185, 40, 39, 37, 167],
    17: [326, 393, 267, 269, 270, 409, 391],
    18: [97, 98, 203, 186, 185, 165],
    19: [326, 391, 409, 410, 423, 327],
    20: [54, 21, 162, 127, 116, 143, 156, 70, 63, 68],
    21: [284, 298, 293, 300, 383, 372, 345, 356, 389, 251],
    22: [126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209],
    23: [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329],
    24: [203, 205, 187, 147, 177, 215, 138, 172, 136, 135, 212, 186, 206],
    25: [423, 426, 410, 432, 364, 365, 397, 367, 435, 401, 376, 411, 425],
    26: [18, 83, 182, 194, 32, 140, 176, 148, 152, 377, 400, 369, 262, 418, 406, 313],
    27: [57, 212, 210, 169, 150, 149, 176, 140, 204, 43],
    28: [287, 273, 424, 369, 400, 378, 379, 394, 430, 432],
}


def find_subjects(root):
    subjects = []
    for dirpath, _, files in os.walk(root):
        if any(name.lower().endswith(VIDEO_EXTS) for name in files):
            subjects.append(dirpath)
    subjects.sort()
    return subjects


def prepare_pair(rppg, ppg, fps):
    ppg = np.asarray(ppg, dtype=np.float64)
    rppg = np.asarray(rppg, dtype=np.float64)

    if len(ppg) == 0 or len(rppg) == 0:
        return None, None

    ppg = np.interp(
        np.linspace(0, len(ppg) - 1, len(rppg)),
        np.arange(len(ppg)),
        ppg,
    )
    rppg = standardize(bandpass(rppg, fps))
    ppg = standardize(bandpass(ppg, fps))

    return rppg, ppg


def extract_all_roi_rgb(video_path, roi_map, face_mesh):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    roi_traces = {roi_id: [] for roi_id in roi_map}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            for roi_id in roi_map:
                roi_traces[roi_id].append([np.nan, np.nan, np.nan])
            continue

        h, w, _ = frame_rgb.shape
        pts = np.array(
            [
                (int(lm.x * w), int(lm.y * h))
                for lm in result.multi_face_landmarks[0].landmark
            ],
            dtype=np.int32,
        )

        for roi_id, roi_idx in roi_map.items():
            rgb = roi_mean_rgb(frame_rgb, pts, roi_idx)
            roi_traces[roi_id].append(rgb if rgb is not None else [np.nan, np.nan, np.nan])

    cap.release()

    for roi_id in roi_map:
        trace = np.asarray(roi_traces[roi_id], dtype=np.float64)
        trace = interpolate_rgb_trace(trace)
        trace = smooth_rgb_trace(trace, window=5)
        roi_traces[roi_id] = trace

    return roi_traces, float(fps)


def evaluate_subject(subject_path, face_mesh, roi_map=ROI_MAP, min_frames=200):
    videos = [name for name in os.listdir(subject_path) if name.lower().endswith(VIDEO_EXTS)]
    if not videos:
        return []

    video_path = os.path.join(subject_path, videos[0])
    ppg = load_ppg(subject_path)
    if ppg is None:
        return []

    roi_traces, fps = extract_all_roi_rgb(video_path, roi_map, face_mesh)
    rows = []

    for roi_id, rgb_trace in roi_traces.items():
        if len(rgb_trace) < min_frames:
            continue

        rppg, method_weights, method_scores = extract_pulse_signal(rgb_trace, fps)
        rppg, ppg_pair = prepare_pair(rppg, ppg, fps)
        if rppg is None or ppg_pair is None:
            continue

        zero_corr = float(np.corrcoef(rppg, ppg_pair)[0, 1])
        best_corr, best_lag = best_lag_correlation(rppg, ppg_pair, fps)

        rows.append(
            {
                "subject": subject_path,
                "roi_id": roi_id,
                "fps": fps,
                "zero_corr": zero_corr,
                "abs_zero_corr": abs(zero_corr),
                "best_corr": float(best_corr),
                "abs_best_corr": abs(float(best_corr)),
                "best_lag_frames": int(best_lag),
                "chrom_weight": float(method_weights[0]),
                "pos_weight": float(method_weights[1]),
                "chrom_score": float(method_scores[0]),
                "pos_score": float(method_scores[1]),
            }
        )

    return rows


def summarize_results(subject_df):
    summary_df = (
        subject_df.groupby("roi_id", as_index=False)
        .agg(
            n_subjects=("subject", "count"),
            mean_zero_corr=("zero_corr", "mean"),
            mean_abs_zero_corr=("abs_zero_corr", "mean"),
            mean_best_corr=("best_corr", "mean"),
            mean_abs_best_corr=("abs_best_corr", "mean"),
            median_best_corr=("best_corr", "median"),
            mean_chrom_weight=("chrom_weight", "mean"),
            mean_pos_weight=("pos_weight", "mean"),
        )
        .sort_values("mean_abs_best_corr", ascending=False)
        .reset_index(drop=True)
    )
    return summary_df


def plot_roi_ranking(summary_df, metric="mean_abs_best_corr", top_k=None, figsize=(14, 7)):
    plot_df = summary_df.copy()
    if top_k is not None:
        plot_df = plot_df.head(top_k)

    plt.figure(figsize=figsize)
    labels = [f"ROI {roi}" for roi in plot_df["roi_id"]]
    plt.bar(labels, plot_df[metric])
    plt.xticks(rotation=60)
    plt.ylabel(metric)
    plt.title(f"ROI ranking by {metric}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_top_roi_distributions(subject_df, summary_df, top_k=5, metric="abs_best_corr"):
    top_rois = summary_df.head(top_k)["roi_id"].tolist()
    plt.figure(figsize=(12, 6))

    data = [subject_df.loc[subject_df["roi_id"] == roi, metric].values for roi in top_rois]
    labels = [f"ROI {roi}" for roi in top_rois]

    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(metric)
    plt.title(f"Top {top_k} ROI distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_roi_benchmark(root, roi_map=ROI_MAP, min_frames=200, csv_prefix="roi_benchmark"):
    subjects = find_subjects(root)
    print("Subjects found:", len(subjects))

    rows = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        for idx, subject_path in enumerate(subjects, start=1):
            print(f"[{idx}/{len(subjects)}] Processing: {subject_path}")
            rows.extend(evaluate_subject(subject_path, face_mesh, roi_map=roi_map, min_frames=min_frames))

    if not rows:
        raise RuntimeError("No ROI benchmark rows were produced. Check ROOT and dataset files.")

    subject_df = pd.DataFrame(rows)
    summary_df = summarize_results(subject_df)

    subject_csv = f"{csv_prefix}_subject_results.csv"
    summary_csv = f"{csv_prefix}_summary.csv"
    subject_df.to_csv(subject_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nTop ROIs by mean absolute best-lag correlation:")
    print(summary_df[["roi_id", "n_subjects", "mean_abs_best_corr", "mean_best_corr"]].head(10))
    print(f"\nSaved: {subject_csv}")
    print(f"Saved: {summary_csv}")

    plot_roi_ranking(summary_df, metric="mean_abs_best_corr")
    plot_top_roi_distributions(subject_df, summary_df, top_k=5, metric="abs_best_corr")

    return subject_df, summary_df

