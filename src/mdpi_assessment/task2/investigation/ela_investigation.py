"""
Generated functions for my specific case study.
Utility to peforming a error level analysis for the aggregated pairs
Reuses Task 1 image processing functions (grayscale, Gaussian blur)
"""

from pathlib import Path
import numpy as np
import cv2
import csv
from PIL import Image, ImageChops, ImageEnhance
from mdpi_assessment.task2.aggregation.candidate_collector import collect_candidates
from mdpi_assessment.config import RAW_DIR
from mdpi_assessment.task1.process_image import to_grayscale, gaussian_blur  # <-- Reuse Task 1

# ============================================================
# ELA CORE FUNCTIONS
# ============================================================

def compute_ela_image(path: Path, quality=95, rescale=10) -> np.ndarray:
    """Compute Error Level Analysis (ELA) image from a JPEG."""
    original = Image.open(path).convert("RGB")
    tmp = path.with_suffix(".ela_tmp.jpg")
    original.save(tmp, "JPEG", quality=quality)
    compressed = Image.open(tmp)
    ela = ImageChops.difference(original, compressed)

    extrema = ela.getextrema()
    max_diff = max(e[1] for e in extrema)
    scale = rescale * 255 / max_diff if max_diff != 0 else 1
    ela = ImageEnhance.Brightness(ela).enhance(scale)

    tmp.unlink(missing_ok=True)
    return np.array(ela)


def ela_gray(ela):
    """Convert ELA image to grayscale using mean over RGB channels."""
    return np.mean(ela, axis=2)


# ============================================================
# ELA++ FORENSIC METRICS
# ============================================================

def ela_uniformity(ela):
    g = ela_gray(ela)
    return float(np.std(g) / (np.mean(g) + 1e-6))


def ela_energy_ratio(ela, percentile=95):
    g = ela_gray(ela)
    thresh = np.percentile(g, percentile)
    return float(np.sum(g > thresh) / g.size)


def ela_spatial_clustering(ela):
    g = ela_gray(ela)
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    return float(np.mean(gx) + np.mean(gy))


def ela_edge_alignment(ela, orig_bgr):
    g_ela = ela_gray(ela).astype(np.uint8)
    gray = to_grayscale(orig_bgr)  # 
    edges = cv2.Canny(gray, 100, 200)

    on = g_ela[edges > 0]
    off = g_ela[edges == 0]
    return float(np.mean(on) / (np.mean(off) + 1e-6))


def ela_hotspot_mask(ela, top_percent=1):
    g = ela_gray(ela)
    thresh = np.percentile(g, 100 - top_percent)
    return g > thresh


# ============================================================
# JPEG & NOISE FORENSICS
# ============================================================

def jpeg_quant_tables(path: Path):
    img = Image.open(path)
    return img.quantization


def quant_table_score(qtables):
    vals = []
    for t in qtables.values():
        vals.extend(t)
    return float(np.mean(vals)) if vals else 0.0


def noise_channel_imbalance(img):
    """Use Task 1 Gaussian blur for residuals"""
    blur = gaussian_blur(img)  # <-- Reuse Task 1 Gaussian blur
    residual = img.astype(np.float32) - blur.astype(np.float32)
    stds = [np.std(residual[:, :, i]) for i in range(3)]
    return float(max(stds) / (min(stds) + 1e-6))


# ============================================================
# FORENSIC DECISION
# ============================================================

def edit_likelihood(ela, orig_bgr):
    return (
        1.0 * ela_uniformity(ela) +
        1.2 * ela_energy_ratio(ela) +
        1.0 * ela_spatial_clustering(ela) +
        1.5 * (1 - ela_edge_alignment(ela, orig_bgr))
    )


def final_forensic_score(ela, orig, qtables):
    return (
        2.0 * edit_likelihood(ela, orig) +
        1.5 * (1 / (quant_table_score(qtables) + 1e-6)) +
        1.0 * noise_channel_imbalance(orig)
    )


def visualize_noise_residual(img):
    blur = gaussian_blur(img)  
    residual = cv2.absdiff(img, blur)
    residual_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
    return residual_norm


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(orig, ela, mask, out_path):
    ela_norm = cv2.normalize(ela_gray(ela), None, 0, 255, cv2.NORM_MINMAX)
    ela_norm = cv2.cvtColor(ela_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    overlay = orig.copy()
    overlay[mask] = [0, 0, 255]

    vis = np.vstack([orig, ela_norm, overlay])
    cv2.imwrite(str(out_path), vis)


def save_composite_forensic_figure_cv(orig_a_path, orig_b_path, ela_a, ela_b, mask_a, mask_b, out_path):
    orig_a = cv2.imread(str(orig_a_path))
    orig_b = cv2.imread(str(orig_b_path))

    ela_a_norm = cv2.normalize(ela_gray(ela_a), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ela_a_bgr = cv2.cvtColor(ela_a_norm, cv2.COLOR_GRAY2BGR)

    ela_b_norm = cv2.normalize(ela_gray(ela_b), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ela_b_bgr = cv2.cvtColor(ela_b_norm, cv2.COLOR_GRAY2BGR)

    hotspot_a = orig_a.copy()
    hotspot_a[mask_a] = [0, 0, 255]
    hotspot_b = orig_b.copy()
    hotspot_b[mask_b] = [0, 0, 255]

    edges_a = cv2.Canny(to_grayscale(orig_a), 100, 200)  # <-- Task 1 grayscale
    edge_a_overlay = cv2.addWeighted(orig_a, 0.7, cv2.cvtColor(edges_a, cv2.COLOR_GRAY2BGR), 0.3, 0)

    edges_b = cv2.Canny(to_grayscale(orig_b), 100, 200)  # <-- Task 1 grayscale
    edge_b_overlay = cv2.addWeighted(orig_b, 0.7, cv2.cvtColor(edges_b, cv2.COLOR_GRAY2BGR), 0.3, 0)

    noise_a = visualize_noise_residual(orig_a)
    noise_b = visualize_noise_residual(orig_b)

    col_a = np.vstack([orig_a, ela_a_bgr, hotspot_a, edge_a_overlay, noise_a])
    col_b = np.vstack([orig_b, ela_b_bgr, hotspot_b, edge_b_overlay, noise_b])

    composite = np.hstack([col_a, col_b])
    cv2.imwrite(str(out_path), composite)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_ela_investigation(results_dir: Path, strategy_csvs: dict, image_dir: Path = None):
    image_dir = image_dir or RAW_DIR
    candidates = collect_candidates(
        [(p, n) for n, p in strategy_csvs.items()],
        min_votes=1
    )

    vis_dir = results_dir / "ela_forensics"
    vis_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, c in enumerate(candidates):
        a_path = image_dir / c["image_a"]
        b_path = image_dir / c["image_b"]

        orig_a = cv2.imread(str(a_path))
        orig_b = cv2.imread(str(b_path))

        ela_a = compute_ela_image(a_path, quality=95, rescale=20)
        ela_b = compute_ela_image(b_path, quality=95, rescale=20)

        q_a = jpeg_quant_tables(a_path)
        q_b = jpeg_quant_tables(b_path)

        score_a = final_forensic_score(ela_a, orig_a, q_a)
        score_b = final_forensic_score(ela_b, orig_b, q_b)

        edited = "image_b" if score_b > score_a else "image_a"

        mask_a = ela_hotspot_mask(ela_a, top_percent=5)
        mask_b = ela_hotspot_mask(ela_b, top_percent=5)

        save_visualization(orig_a, ela_a, mask_a, vis_dir / f"{idx}_A_{a_path.name}")
        save_visualization(orig_b, ela_b, mask_b, vis_dir / f"{idx}_B_{b_path.name}")

        results.append({
            "image_a": c["image_a"],
            "image_b": c["image_b"],
            "votes": c["vote_count"],
            "sources": ",".join(c["sources"]),
            "max_score": c["max_score"],
            "forensic_score_a": score_a,
            "forensic_score_b": score_b,
            "edited_image": edited,
            "ela_uniformity_a": ela_uniformity(ela_a),
            "ela_uniformity_b": ela_uniformity(ela_b),
            "ela_energy_a": ela_energy_ratio(ela_a),
            "ela_energy_b": ela_energy_ratio(ela_b),
            "ela_cluster_a": ela_spatial_clustering(ela_a),
            "ela_cluster_b": ela_spatial_clustering(ela_b),
            "ela_edge_align_a": ela_edge_alignment(ela_a, orig_a),
            "ela_edge_align_b": ela_edge_alignment(ela_b, orig_b),
            "quant_mean_a": quant_table_score(q_a),
            "quant_mean_b": quant_table_score(q_b),
            "noise_imbalance_a": noise_channel_imbalance(orig_a),
            "noise_imbalance_b": noise_channel_imbalance(orig_b),
            "ela_a": ela_a,
            "ela_b": ela_b,
            "mask_a": mask_a,
            "mask_b": mask_b,
            "a_path": a_path,
            "b_path": b_path,
        })

    for row in results:
        for key in ["forensic_score_a","forensic_score_b","ela_uniformity_a","ela_uniformity_b",
                    "ela_energy_a","ela_energy_b","ela_cluster_a","ela_cluster_b",
                    "ela_edge_align_a","ela_edge_align_b","quant_mean_a","quant_mean_b",
                    "noise_imbalance_a","noise_imbalance_b"]:
            row[key] = f"{row[key]:.4f}" if "ela" in key else f"{row[key]:.3f}"

    filtered = [r for r in results if int(r["votes"]) >= 2]
    best_pair = max(filtered, key=lambda r: min(float(r["forensic_score_a"]), float(r["forensic_score_b"])))

    composite_path = vis_dir / f"best_pair_{best_pair['image_a']}_{best_pair['image_b']}.png"
    save_composite_forensic_figure_cv(
        best_pair["a_path"],
        best_pair["b_path"],
        best_pair["ela_a"],
        best_pair["ela_b"],
        best_pair["mask_a"],
        best_pair["mask_b"],
        composite_path
    )

    for r in results:
        for k in ["ela_a","ela_b","mask_a","mask_b","a_path","b_path"]:
            r.pop(k, None)

    out_csv = results_dir / "task2_ela_forensics.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"[OK] Forensic ELA results saved to {out_csv}")
    print(f"[OK] Composite figure for best candidate saved to {composite_path}")
    return results