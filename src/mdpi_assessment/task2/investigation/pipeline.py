from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import cv2

from mdpi_assessment.config import RAW_DIR
from mdpi_assessment.logger import logger
from mdpi_assessment.task2.aggregation.candidate_collector import collect_candidates

from .core import compute_ela_image, jpeg_quantization_tables
from .metrics import (
    ela_edge_alignment,
    ela_energy_ratio,
    ela_hotspot_mask,
    ela_spatial_clustering,
    ela_uniformity,
    noise_channel_imbalance,
    quantization_table_score,
)
from .scoring import final_forensic_score
from .viz import save_composite_forensic_figure_cv, save_visualization


# number of votes from different strategy,
# 1 -> allows all to vote, 2 -> at least two strategies have to agree
def run_ela_investigation(
    results_directory: Path,
    strategy_csv_paths: Dict[str, Path],
    image_directory: Path | None = None,
    min_votes: int = 2,
) -> List[Dict]:
    image_directory = image_directory or RAW_DIR

    candidates = collect_candidates(
        [
            (csv_path, strategy_name)
            for strategy_name, csv_path in strategy_csv_paths.items()
        ],
        min_votes=min_votes,
    )

    visualization_directory = results_directory / "ela_forensics"
    visualization_directory.mkdir(parents=True, exist_ok=True)

    investigation_results: List[Dict] = []

    for candidate_index, candidate in enumerate(candidates):
        image_a_path = image_directory / candidate["image_a"]
        image_b_path = image_directory / candidate["image_b"]

        original_image_a_bgr = cv2.imread(str(image_a_path))
        original_image_b_bgr = cv2.imread(str(image_b_path))

        if original_image_a_bgr is None or original_image_b_bgr is None:
            logger.warning("Could not read images %s or %s", image_a_path, image_b_path)
            continue

        ela_image_a_rgb = compute_ela_image(image_a_path, quality=95, rescale=20)
        ela_image_b_rgb = compute_ela_image(image_b_path, quality=95, rescale=20)

        quant_tables_a = jpeg_quantization_tables(image_a_path)
        quant_tables_b = jpeg_quantization_tables(image_b_path)

        forensic_score_a = final_forensic_score(
            ela_image_a_rgb, original_image_a_bgr, quant_tables_a
        )
        forensic_score_b = final_forensic_score(
            ela_image_b_rgb, original_image_b_bgr, quant_tables_b
        )

        edited_image_label = (
            "image_b" if forensic_score_b > forensic_score_a else "image_a"
        )

        hotspot_mask_a = ela_hotspot_mask(ela_image_a_rgb, top_percent=5.0)
        hotspot_mask_b = ela_hotspot_mask(ela_image_b_rgb, top_percent=5.0)

        save_visualization(
            original_image_a_bgr,
            ela_image_a_rgb,
            hotspot_mask_a,
            visualization_directory / f"{candidate_index}_A_{image_a_path.name}",
        )
        save_visualization(
            original_image_b_bgr,
            ela_image_b_rgb,
            hotspot_mask_b,
            visualization_directory / f"{candidate_index}_B_{image_b_path.name}",
        )

        investigation_results.append(
            {
                "image_a": candidate["image_a"],
                "image_b": candidate["image_b"],
                "votes": candidate["vote_count"],
                "sources": ",".join(candidate["sources"]),
                "max_score": candidate["max_score"],
                "forensic_score_a": f"{forensic_score_a:.4f}",
                "forensic_score_b": f"{forensic_score_b:.4f}",
                "edited_image": edited_image_label,
                "ela_uniformity_a": f"{ela_uniformity(ela_image_a_rgb):.4f}",
                "ela_uniformity_b": f"{ela_uniformity(ela_image_b_rgb):.4f}",
                "ela_energy_a": f"{ela_energy_ratio(ela_image_a_rgb):.4f}",
                "ela_energy_b": f"{ela_energy_ratio(ela_image_b_rgb):.4f}",
                "ela_cluster_a": f"{ela_spatial_clustering(ela_image_a_rgb):.4f}",
                "ela_cluster_b": f"{ela_spatial_clustering(ela_image_b_rgb):.4f}",
                "ela_edge_align_a": f"{ela_edge_alignment(ela_image_a_rgb, original_image_a_bgr):.4f}",
                "ela_edge_align_b": f"{ela_edge_alignment(ela_image_b_rgb, original_image_b_bgr):.4f}",
                "quant_mean_a": f"{quantization_table_score(quant_tables_a):.3f}",
                "quant_mean_b": f"{quantization_table_score(quant_tables_b):.3f}",
                "noise_imbalance_a": f"{noise_channel_imbalance(original_image_a_bgr):.3f}",
                "noise_imbalance_b": f"{noise_channel_imbalance(original_image_b_bgr):.3f}",
                "ela_image_a_rgb": ela_image_a_rgb,
                "ela_image_b_rgb": ela_image_b_rgb,
                "hotspot_mask_a": hotspot_mask_a,
                "hotspot_mask_b": hotspot_mask_b,
                "image_a_path": image_a_path,
                "image_b_path": image_b_path,
            }
        )

    if not investigation_results:
        logger.warning("No ELA investigation results produced.")
        return []

    filtered_results = [
        result for result in investigation_results if int(result["votes"]) >= min_votes
    ]
    if not filtered_results:
        filtered_results = investigation_results

    best_pair = max(
        filtered_results,
        key=lambda result: min(
            float(result["forensic_score_a"]),
            float(result["forensic_score_b"]),
        ),
    )

    composite_path = (
        visualization_directory
        / f"best_pair_{best_pair['image_a']}_{best_pair['image_b']}.png"
    )
    save_composite_forensic_figure_cv(
        best_pair["image_a_path"],
        best_pair["image_b_path"],
        best_pair["ela_image_a_rgb"],
        best_pair["ela_image_b_rgb"],
        best_pair["hotspot_mask_a"],
        best_pair["hotspot_mask_b"],
        composite_path,
    )

    for result in investigation_results:
        for key in (
            "ela_image_a_rgb",
            "ela_image_b_rgb",
            "hotspot_mask_a",
            "hotspot_mask_b",
            "image_a_path",
            "image_b_path",
        ):
            result.pop(key, None)

    output_csv_path = results_directory / "task2_ela_forensics.csv"
    with output_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=investigation_results[0].keys())
        writer.writeheader()
        writer.writerows(investigation_results)

    logger.info("Forensic ELA results saved to %s", output_csv_path)
    logger.info("Composite figure for best candidate saved to %s", composite_path)

    return investigation_results
