"""
Ground-truth-guided ROI-4 refiner.

Important:
- This file uses PPG during selection/fusion to maximize correlation.
- It is useful for debugging and estimating the upper bound of ROI 4.
- It is NOT a fair deployment pipeline for later HR prediction.

Usage:

    from roi4_gt_guided_refiner import process_dataset_roi4_gt, plot_subject_gt_result

    results = process_dataset_roi4_gt(ROOT, top_k=5)
    plot_subject_gt_result(results, list(results.keys())[0])
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from roi4_rppg_advanced import (
    VIDEO_EXTS,
    bandpass,
    best_lag_correlation,
    build_patch_candidate,
    extract_roi4_patch_traces,
    find_subjects,
    load_ppg,
    mp_face_mesh,
    prepare_pair,
    standardize,
)


def align_by_lag(reference, signal, lag):
    reference = np.asarray(reference, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if lag < 0:
        x = reference[-lag:]
        y = signal[: len(x)]
    elif lag > 0:
        x = reference[:-lag]
        y = signal[lag:]
    else:
        x = reference
        y = signal

    n = min(len(x), len(y))
    return x[:n], y[:n]


def score_candidate_vs_ppg(candidate_signal, ppg, fps):
    rppg, ppg = prepare_pair(candidate_signal, ppg, fps)
    if rppg is None or ppg is None:
        return None

    zero_corr = float(np.corrcoef(rppg, ppg)[0, 1])
    best_corr, best_lag = best_lag_correlation(rppg, ppg, fps)
    abs_best = abs(float(best_corr))

    return {
        "rppg": rppg,
        "ppg": ppg,
        "zero_corr": zero_corr,
        "best_corr": float(best_corr),
        "abs_best_corr": abs_best,
        "best_lag": int(best_lag),
    }


def evaluate_subject_roi4_gt(subject_path, face_mesh, top_k=5, min_frames=200, grid_rows=3, grid_cols=3):
    videos = [name for name in os.listdir(subject_path) if name.lower().endswith(VIDEO_EXTS)]
    if not videos:
        return None

    ppg = load_ppg(subject_path)
    if ppg is None:
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

    candidates = []
    for patch_idx, trace in enumerate(patch_traces):
        signal, _, _, best_method, candidate_dict = build_patch_candidate(trace, fps)

        # Evaluate all candidate methods, not just the unsupervised winner.
        for method_name, info in candidate_dict.items():
            scored = score_candidate_vs_ppg(info["signal"], ppg, fps)
            if scored is None:
                continue

            candidates.append(
                {
                    "patch": patch_idx,
                    "method": method_name,
                    "signal": scored["rppg"],
                    "ppg": scored["ppg"],
                    "zero_corr": scored["zero_corr"],
                    "best_corr": scored["best_corr"],
                    "abs_best_corr": scored["abs_best_corr"],
                    "best_lag": scored["best_lag"],
                }
            )

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["abs_best_corr"], reverse=True)
    top = candidates[: max(1, min(top_k, len(candidates)))]

    ref = top[0]["signal"]
    fused = np.zeros_like(ref, dtype=np.float64)
    weights = np.array([item["abs_best_corr"] for item in top], dtype=np.float64)
    if np.sum(weights) <= 0:
        weights = np.ones(len(top), dtype=np.float64)
    weights = weights / np.sum(weights)

    aligned_info = []
    for weight, item in zip(weights, top):
        sig = item["signal"]

        # First align polarity to reference.
        corr_ref = np.corrcoef(ref, sig)[0, 1]
        if np.isfinite(corr_ref) and corr_ref < 0:
            sig = -sig

        # Then align by the best lag found against PPG.
        _, sig_aligned = align_by_lag(ref, sig, item["best_lag"])
        ref_aligned, _ = align_by_lag(ref, sig, item["best_lag"])

        n = min(len(ref_aligned), len(sig_aligned), len(fused))
        fused[:n] += weight * sig_aligned[:n]

        aligned_info.append(
            {
                "patch": item["patch"],
                "method": item["method"],
                "weight": float(weight),
                "abs_best_corr": float(item["abs_best_corr"]),
            }
        )

    fused = standardize(bandpass(fused, fps))
    fused, ppg_pair = prepare_pair(fused, ppg, fps)
    zero_corr = float(np.corrcoef(fused, ppg_pair)[0, 1])
    best_corr, best_lag = best_lag_correlation(fused, ppg_pair, fps)

    return {
        "subject": subject_path,
        "fps": fps,
        "rppg": fused,
        "ppg": ppg_pair,
        "zero_corr": zero_corr,
        "best_corr": float(best_corr),
        "best_lag": int(best_lag),
        "top_candidates": aligned_info,
        "all_candidates": [
            {
                "patch": item["patch"],
                "method": item["method"],
                "abs_best_corr": float(item["abs_best_corr"]),
                "best_corr": float(item["best_corr"]),
            }
            for item in candidates
        ],
    }


def process_dataset_roi4_gt(root, top_k=5, min_frames=200, grid_rows=3, grid_cols=3):
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
            result = evaluate_subject_roi4_gt(
                subj,
                face_mesh,
                top_k=top_k,
                min_frames=min_frames,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
            )
            if result is not None:
                results[subj] = result

    return results


def plot_subject_gt_result(results, subject_key):
    data = results[subject_key]
    fps = data["fps"]
    rppg = data["rppg"]
    ppg = data["ppg"]
    t = np.arange(len(rppg)) / fps

    plt.figure(figsize=(12, 4))
    plt.plot(t, rppg, label="GT-guided ROI4 rPPG")
    plt.plot(t, ppg, label="PPG", alpha=0.8)
    plt.title(subject_key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Zero-lag correlation:", data["zero_corr"])
    print("Best correlation:", data["best_corr"])
    print("Best lag (frames):", data["best_lag"])
    print("Top candidates:")
    for item in data["top_candidates"]:
        print(item)
