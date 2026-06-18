#!/usr/bin/env python3
"""Derm Immune Profiler.

Score inflammatory skin biopsy expression profiles against the immune-module
signature from Seremet et al. / PersoMed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SIGNATURE_PATH = SCRIPT_DIR / "data" / "der_signature.csv"
PUBLIC_CONTEXT_PATH = SCRIPT_DIR / "data" / "public_geo_context.csv"
DEMO_EXPRESSION = SCRIPT_DIR / "example_data" / "demo_expression.csv"

MODULE_ORDER = ["Th1", "Th2", "Th17", "Neutro", "Macro", "Eosino", "IFN"]
DISPLAY_MODULE_ORDER = ["Th2", "Th17", "Neutro", "Th1", "Eosino", "IFN", "Macro"]
PUBLIC_GROUP_ORDER = [
    "Healthy",
    "AD",
    "PsO",
    "LP",
    "CLE",
    "NeuD",
    "Wells",
    "BP",
    "DHR",
    "Erythroderma",
    "Undetermined rash",
    "COVID skin lesion",
]
MODULE_COLORS = {
    "Th1": "#c77c2c",
    "Th2": "#4f87c7",
    "Th17": "#69a84f",
    "Neutro": "#c46ac4",
    "Macro": "#c94f4f",
    "Eosino": "#7b63c6",
    "IFN": "#d6b83f",
}
REFERENCE_GROUPS = [
    ("AD", "Atopic Dermatitis", "#f8766d"),
    ("BP", "Bullous Pemphigoid", "#d59b00"),
    ("CLE", "Cutaneous Lupus Erythematosus", "#7cae00"),
    ("DHR", "Drug Hypersensitivity Reaction", "#00bf7d"),
    ("Healthy", "Healthy Donor", "#00bfc4"),
    ("LP", "Lichen Planus", "#00a9ff"),
    ("PsO", "Plaque Psoriasis", "#ff61c3"),
    ("NeuD", "Neutrophilic Dermatoses", "#c77cff"),
    ("Wells", "Wells syndrome", "#8a8a8a"),
    ("Erythroderma", "Erythroderma", "#8c564b"),
    ("Undetermined rash", "Undetermined rash", "#6b7280"),
    ("COVID skin lesion", "COVID-19 skin lesion", "#444444"),
]
GROUP_COLORS = {group: color for group, _, color in REFERENCE_GROUPS}
GROUP_LABELS = {group: label for group, label, _ in REFERENCE_GROUPS}
EMBEDDING_FEATURES = MODULE_ORDER
EMBEDDING_K = 7
THRESHOLDS = {
    "Th1": 0.56,
    "Th2": 0.33,
    "Th17": 0.44,
    "Neutro": 0.53,
    "Macro": 0.50,
    "Eosino": 1.05,
    "IFN": 0.86,
}

MODULE_INTERPRETATION = {
    "Th17": {
        "sentinel": "psoriasis-like / IL-17-IL-23-axis inflammation",
        "target_frame": "anti-IL-17A/F or anti-IL-23 pathway alignment in the paper",
    },
    "Th2": {
        "sentinel": "atopic-dermatitis-like / type-2 inflammation",
        "target_frame": "anti-IL-4RA or anti-IL-13 pathway alignment in the paper",
    },
    "Th1": {
        "sentinel": "lichen-planus-like / Th1 cytotoxic inflammation",
        "target_frame": "JAK/STAT or Th1-axis pathway alignment discussed in the paper",
    },
    "IFN": {
        "sentinel": "cutaneous-lupus-like / type-I-interferon inflammation",
        "target_frame": "type-I-IFN/JAK pathway alignment discussed in the paper",
    },
    "Neutro": {
        "sentinel": "neutrophilic-dermatosis-like inflammation",
        "target_frame": "IL-1/IL-36 neutrophilic pathway alignment discussed in the paper",
    },
    "Macro": {
        "sentinel": "macrophage-associated myeloid inflammation",
        "target_frame": "myeloid pathway signal; treatment mapping is exploratory",
    },
    "Eosino": {
        "sentinel": "eosinophilic/Wells-like inflammation",
        "target_frame": "eosinophilic pathway signal; paper notes limited discriminatory power",
    },
}

DISCLAIMER = (
    "ClawBio is a research and educational tool. It is not a medical device "
    "and does not provide clinical diagnoses. Consult a healthcare professional "
    "before making any medical decisions."
)


@dataclass(frozen=True)
class ModuleScore:
    module: str
    genes_present: int
    genes_expected: int
    raw_mean: float
    threshold: float
    activation: float


@dataclass(frozen=True)
class SampleProfile:
    sample_id: str
    scores: list[ModuleScore]
    dominant_module: str
    codominant_modules: list[str]
    missing_genes: list[str]


@dataclass(frozen=True)
class Neighbor:
    source_accession: str
    sample_id: str
    group: str
    distance: float
    weight: float


@dataclass(frozen=True)
class PublicEmbeddingPoint:
    source_accession: str
    sample_id: str
    group: str
    x: float
    y: float


@dataclass(frozen=True)
class QueryEmbeddingPoint:
    sample_id: str
    x: float
    y: float
    neighbors: list[Neighbor]


@dataclass(frozen=True)
class ReferenceEmbedding:
    method: str
    features: list[str]
    k_neighbors: int
    public_points: list[PublicEmbeddingPoint]
    query_points: list[QueryEmbeddingPoint]
    bounds: tuple[float, float, float, float]


def svg_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def detect_delimiter(path: Path) -> str:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"
    return ","


def load_signature(path: Path = SIGNATURE_PATH) -> dict[str, list[str]]:
    modules = {module: [] for module in MODULE_ORDER}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gene = row["genes"].strip()
            module = row["signature"].strip()
            if module not in modules:
                modules[module] = []
            modules[module].append(gene)
    return modules


def load_expression(path: Path) -> dict[str, dict[str, float]]:
    delimiter = detect_delimiter(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Input expression table has no header row")
        if "sample_id" not in reader.fieldnames:
            raise ValueError("Input expression table must include a 'sample_id' column")
        samples: dict[str, dict[str, float]] = {}
        for row in reader:
            if None in row:
                raise ValueError("Input expression table has rows with more values than header columns")
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError("Every row must include a non-empty sample_id")
            values: dict[str, float] = {}
            for key, value in row.items():
                if key == "sample_id" or value is None or value == "":
                    continue
                try:
                    values[key.strip()] = float(value)
                except ValueError as exc:
                    raise ValueError(f"Non-numeric expression value for {sample_id}/{key}: {value}") from exc
            samples[sample_id] = values
    if not samples:
        raise ValueError("Input expression table contains no samples")
    return samples


def logistic_activation(raw_mean: float, threshold: float) -> float:
    return 1.0 / (1.0 + math.exp(-3.0 * (raw_mean - threshold)))


def score_sample(sample_id: str, values: dict[str, float], modules: dict[str, list[str]]) -> SampleProfile:
    scores: list[ModuleScore] = []
    missing: list[str] = []
    for module in MODULE_ORDER:
        genes = modules[module]
        present_values = [values[gene] for gene in genes if gene in values]
        missing.extend(gene for gene in genes if gene not in values)
        if not present_values:
            raw_mean = float("nan")
            activation = 0.0
        else:
            raw_mean = sum(present_values) / len(present_values)
            activation = logistic_activation(raw_mean, THRESHOLDS[module])
        scores.append(
            ModuleScore(
                module=module,
                genes_present=len(present_values),
                genes_expected=len(genes),
                raw_mean=raw_mean,
                threshold=THRESHOLDS[module],
                activation=activation,
            )
        )

    dominant = max(scores, key=lambda score: score.activation)
    cutoff = max(0.7 * dominant.activation, 0.30)
    codominant = [score.module for score in scores if score.activation > cutoff]
    return SampleProfile(
        sample_id=sample_id,
        scores=scores,
        dominant_module=dominant.module,
        codominant_modules=codominant,
        missing_genes=sorted(set(missing)),
    )


def score_expression_table(input_path: Path) -> list[SampleProfile]:
    modules = load_signature()
    samples = load_expression(input_path)
    return [score_sample(sample_id, values, modules) for sample_id, values in samples.items()]


def load_public_context(path: Path = PUBLIC_CONTEXT_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for module in MODULE_ORDER:
                parsed[f"{module}_activation"] = float(parsed[f"{module}_activation"])
                raw_value = parsed.get(f"{module}_raw_mean", "")
                parsed[f"{module}_raw_mean"] = None if raw_value in {"", "None"} else float(raw_value)
            rows.append(parsed)
        return rows


def profile_to_record(profile: SampleProfile) -> dict:
    score_records = []
    for score in profile.scores:
        raw_mean = None if math.isnan(score.raw_mean) else round(score.raw_mean, 4)
        score_records.append(
            {
                "module": score.module,
                "raw_mean": raw_mean,
                "threshold": score.threshold,
                "activation": round(score.activation, 4),
                "genes_present": score.genes_present,
                "genes_expected": score.genes_expected,
            }
        )
    dominant_info = MODULE_INTERPRETATION[profile.dominant_module]
    return {
        "sample_id": profile.sample_id,
        "dominant_module": profile.dominant_module,
        "codominant_modules": profile.codominant_modules,
        "interpretation": dominant_info["sentinel"],
        "treatment_target_frame": dominant_info["target_frame"],
        "scores": score_records,
        "missing_signature_genes": profile.missing_genes,
    }


def write_tables(profiles: list[SampleProfile], output_dir: Path, embedding: ReferenceEmbedding | None = None) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    with (tables_dir / "module_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_id",
            "module",
            "raw_mean",
            "threshold",
            "activation",
            "genes_present",
            "genes_expected",
            "dominant_module",
            "codominant_modules",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile in profiles:
            for score in profile.scores:
                writer.writerow(
                    {
                        "sample_id": profile.sample_id,
                        "module": score.module,
                        "raw_mean": "" if math.isnan(score.raw_mean) else f"{score.raw_mean:.6f}",
                        "threshold": f"{score.threshold:.2f}",
                        "activation": f"{score.activation:.6f}",
                        "genes_present": score.genes_present,
                        "genes_expected": score.genes_expected,
                        "dominant_module": profile.dominant_module,
                        "codominant_modules": ";".join(profile.codominant_modules),
                    }
                )

    with (tables_dir / "sample_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "dominant_module",
                "codominant_modules",
                "interpretation",
                "treatment_target_frame",
                "missing_signature_gene_count",
            ],
        )
        writer.writeheader()
        for profile in profiles:
            info = MODULE_INTERPRETATION[profile.dominant_module]
            writer.writerow(
                {
                    "sample_id": profile.sample_id,
                    "dominant_module": profile.dominant_module,
                    "codominant_modules": ";".join(profile.codominant_modules),
                    "interpretation": info["sentinel"],
                    "treatment_target_frame": info["target_frame"],
                    "missing_signature_gene_count": len(profile.missing_genes),
                }
            )

    if embedding is not None:
        with (tables_dir / "embedding_neighbors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "embedding_1",
                    "embedding_2",
                    "neighbor_rank",
                    "neighbor_source_accession",
                    "neighbor_sample_id",
                    "neighbor_group",
                    "distance",
                    "weight",
                ],
            )
            writer.writeheader()
            for point in embedding.query_points:
                for rank, neighbor in enumerate(point.neighbors, start=1):
                    writer.writerow(
                        {
                            "sample_id": point.sample_id,
                            "embedding_1": f"{point.x:.6f}",
                            "embedding_2": f"{point.y:.6f}",
                            "neighbor_rank": rank,
                            "neighbor_source_accession": neighbor.source_accession,
                            "neighbor_sample_id": neighbor.sample_id,
                            "neighbor_group": neighbor.group,
                            "distance": f"{neighbor.distance:.6f}",
                            "weight": f"{neighbor.weight:.6f}",
                        }
                    )


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def profile_activation(profile: SampleProfile, module: str) -> float:
    return _score_map(profile)[module].activation


def grouped_public_context(public_context: list[dict]) -> list[str]:
    present = {row["group"] for row in public_context}
    ordered = [group for group in PUBLIC_GROUP_ORDER if group in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def public_activation_vector(row: dict) -> np.ndarray:
    return np.array([float(row[f"{module}_activation"]) for module in EMBEDDING_FEATURES], dtype=float)


def profile_activation_vector(profile: SampleProfile) -> np.ndarray:
    score_map = _score_map(profile)
    return np.array([score_map[module].activation for module in EMBEDDING_FEATURES], dtype=float)


def _median_coordinate(points: np.ndarray, rows: list[dict], group: str, axis: int) -> float | None:
    values = [float(points[idx, axis]) for idx, row in enumerate(rows) if row.get("group") == group]
    if not values:
        return None
    return median(values)


def embedding_orientation_signs(coords: np.ndarray, rows: list[dict]) -> tuple[float, float]:
    x_sign = 1.0
    y_sign = 1.0
    pso_x = _median_coordinate(coords, rows, "PsO", 0)
    cle_x = _median_coordinate(coords, rows, "CLE", 0)
    if pso_x is not None and cle_x is not None and pso_x < cle_x:
        x_sign = -1.0

    ad_y = _median_coordinate(coords, rows, "AD", 1)
    bp_y = _median_coordinate(coords, rows, "BP", 1)
    if ad_y is not None and bp_y is not None and ad_y < bp_y:
        y_sign = -1.0
    return x_sign, y_sign


def apply_embedding_orientation(coords: np.ndarray, signs: tuple[float, float]) -> np.ndarray:
    oriented = coords.copy()
    oriented[:, 0] *= signs[0]
    oriented[:, 1] *= signs[1]
    return oriented


def embedding_bounds(public_points: list[PublicEmbeddingPoint], query_points: list[QueryEmbeddingPoint]) -> tuple[float, float, float, float]:
    xs = [point.x for point in public_points] + [point.x for point in query_points]
    ys = [point.y for point in public_points] + [point.y for point in query_points]
    if not xs or not ys:
        return -1.0, 1.0, -1.0, 1.0
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max((max_x - min_x) * 0.08, 0.5)
    pad_y = max((max_y - min_y) * 0.08, 0.5)
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def build_reference_embedding(public_context: list[dict], profiles: list[SampleProfile], k_neighbors: int = EMBEDDING_K) -> ReferenceEmbedding | None:
    if not public_context:
        return None

    matrix = np.vstack([public_activation_vector(row) for row in public_context])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0.0] = 1.0
    standardized = (matrix - mean) / std

    method = "source_data_umap_2d_with_umap_transform_and_knn_neighbors"
    query_matrix = np.vstack([profile_activation_vector(profile) for profile in profiles]) if profiles else np.empty((0, len(EMBEDDING_FEATURES)))
    query_standardized = (query_matrix - mean) / std if len(query_matrix) else query_matrix
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "clawbio_numba_cache"))
    os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "numba.core.caching.UserWideCacheLocator")
    try:
        import numba

        original_njit = numba.njit
        original_vectorize = numba.vectorize
        original_guvectorize = numba.guvectorize

        def no_cache_njit(*args, **kwargs):
            kwargs.pop("cache", None)
            return original_njit(*args, **kwargs)

        def no_cache_vectorize(*args, **kwargs):
            kwargs.pop("cache", None)
            return original_vectorize(*args, **kwargs)

        def no_cache_guvectorize(*args, **kwargs):
            kwargs.pop("cache", None)
            return original_guvectorize(*args, **kwargs)

        numba.njit = no_cache_njit
        numba.vectorize = no_cache_vectorize
        numba.guvectorize = no_cache_guvectorize
        try:
            import umap
        finally:
            numba.njit = original_njit
            numba.vectorize = original_vectorize
            numba.guvectorize = original_guvectorize
    except ImportError:
        umap = None

    if umap is not None:
        reducer = umap.UMAP(
            n_neighbors=min(15, len(public_context) - 1),
            n_components=2,
            metric="euclidean",
            random_state=42,
            transform_seed=42,
            min_dist=0.15,
        )
        coords = reducer.fit_transform(standardized)
        query_coords = reducer.transform(query_standardized) if len(query_standardized) else np.empty((0, 2))
    else:
        method = "source_data_pca_2d_with_knn_barycentric_query_projection"
        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        coords = standardized @ vt[:2].T
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
        query_coords = np.empty((0, 2))

    signs = embedding_orientation_signs(coords, public_context)
    coords = apply_embedding_orientation(coords, signs)
    if len(query_standardized):
        if method.startswith("source_data_pca"):
            query_coords = []
        else:
            query_coords = apply_embedding_orientation(query_coords, signs)

    public_points = [
        PublicEmbeddingPoint(
            source_accession=row["source_accession"],
            sample_id=row["sample_id"],
            group=row["group"],
            x=float(coords[idx, 0]),
            y=float(coords[idx, 1]),
        )
        for idx, row in enumerate(public_context)
    ]

    query_points: list[QueryEmbeddingPoint] = []
    k = min(k_neighbors, len(public_context))
    for profile_idx, profile in enumerate(profiles):
        query_vector = query_standardized[profile_idx]
        distances = np.linalg.norm(standardized - query_vector, axis=1)
        nearest_idx = np.argsort(distances)[:k]
        if distances[nearest_idx[0]] == 0:
            weights = np.zeros(k)
            weights[0] = 1.0
        else:
            weights = 1.0 / np.maximum(distances[nearest_idx], 1e-9)
            weights = weights / weights.sum()
        if method.startswith("source_data_pca"):
            query_coord = np.average(coords[nearest_idx], axis=0, weights=weights)
        else:
            query_coord = query_coords[profile_idx]
        neighbors = [
            Neighbor(
                source_accession=public_context[int(idx)]["source_accession"],
                sample_id=public_context[int(idx)]["sample_id"],
                group=public_context[int(idx)]["group"],
                distance=float(distances[int(idx)]),
                weight=float(weights[pos]),
            )
            for pos, idx in enumerate(nearest_idx)
        ]
        query_points.append(
            QueryEmbeddingPoint(
                sample_id=profile.sample_id,
                x=float(query_coord[0]),
                y=float(query_coord[1]),
                neighbors=neighbors,
            )
        )

    return ReferenceEmbedding(
        method=method,
        features=list(EMBEDDING_FEATURES),
        k_neighbors=k,
        public_points=public_points,
        query_points=query_points,
        bounds=embedding_bounds(public_points, query_points),
    )


def embedding_to_record(embedding: ReferenceEmbedding | None) -> dict:
    if embedding is None:
        return {
            "method": "not_available",
            "features": list(EMBEDDING_FEATURES),
            "public_reference_sample_count": 0,
            "query_projection": [],
        }
    return {
        "method": embedding.method,
        "features": embedding.features,
        "k_neighbors": embedding.k_neighbors,
        "public_reference_sample_count": len(embedding.public_points),
        "coordinate_source": (
            "2D UMAP coordinates fitted only from public GEO seven-module activation scores; input samples projected with the fitted UMAP transform. "
            "Nearest public neighbors are computed in the same standardized seven-module score space."
            if embedding.method.startswith("source_data_umap")
            else "2D PCA coordinates fitted only from public GEO seven-module activation scores; input samples projected by inverse-distance KNN barycentric placement in the same standardized seven-module space."
        ),
        "query_projection": [
            {
                "sample_id": point.sample_id,
                "embedding_1": round(point.x, 6),
                "embedding_2": round(point.y, 6),
                "nearest_public_neighbors": [
                    {
                        "source_accession": neighbor.source_accession,
                        "sample_id": neighbor.sample_id,
                        "group": neighbor.group,
                        "distance": round(neighbor.distance, 6),
                        "weight": round(neighbor.weight, 6),
                    }
                    for neighbor in point.neighbors
                ],
            }
            for point in embedding.query_points
        ],
    }


def write_public_context_figures(
    profiles: list[SampleProfile],
    output_dir: Path,
    public_context: list[dict],
    embedding: ReferenceEmbedding | None,
) -> list[dict]:
    if not public_context:
        return []
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    groups = grouped_public_context(public_context)

    dominant_path = figures_dir / "public_geo_dominant_modules.svg"
    heatmap_path = figures_dir / "public_geo_module_heatmap.svg"
    overlay_path = figures_dir / "input_vs_public_geo_context.svg"
    embedding_path = figures_dir / "sentinel_panel_source_embedding.svg"

    if embedding is not None:
        write_sentinel_embedding_svg(embedding_path, public_context, embedding)
    write_dominant_distribution_svg(dominant_path, public_context, groups)
    write_public_heatmap_svg(heatmap_path, public_context, groups)
    write_input_overlay_svg(overlay_path, profiles, public_context)

    artifacts = [
        {"path": "figures/public_geo_dominant_modules.svg", "label": "Public GEO dominant-module distribution"},
        {"path": "figures/public_geo_module_heatmap.svg", "label": "Public GEO median module activation heatmap"},
        {"path": "figures/input_vs_public_geo_context.svg", "label": "Input samples over public GEO activation background"},
    ]
    if embedding is not None:
        artifacts.insert(
            0,
            {
                "path": "figures/sentinel_panel_source_embedding.svg",
                "label": "Source-data UMAP immune map"
                if embedding.method.startswith("source_data_umap")
                else "Source-data KNN immune-map projection",
            },
        )
    return artifacts


def map_embedding_point(
    x_value: float,
    y_value: float,
    bounds: tuple[float, float, float, float],
    plot_x: float,
    plot_y: float,
    plot_w: float,
    plot_h: float,
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0
    x = plot_x + ((x_value - min_x) / (max_x - min_x)) * plot_w
    y = plot_y + plot_h - ((y_value - min_y) / (max_y - min_y)) * plot_h
    return x, y


def write_sentinel_embedding_svg(path: Path, public_context: list[dict], embedding: ReferenceEmbedding) -> None:
    width = 1320
    height = 560
    plot_x = 72
    plot_y = 78
    plot_w = 690
    plot_h = 390
    legend_x = 890
    title = "Source-data UMAP immune map" if embedding.method.startswith("source_data_umap") else "Source-data immune-map projection"
    subtitle = (
        "Public GEO UMAP embedding; query samples are UMAP-transformed red triangles"
        if embedding.method.startswith("source_data_umap")
        else "Public GEO reference embedding; query samples are KNN-projected red triangles"
    )
    footer = (
        "Reference UMAP is fitted from public GEO seven-module activation scores; nearest neighbors are computed in the same score space."
        if embedding.method.startswith("source_data_umap")
        else "Reference coordinates are fitted from public GEO seven-module activation scores; query triangles are inverse-distance KNN projections."
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="72" y="42" font-family="Helvetica,Arial,sans-serif" font-size="24" font-weight="700">{title}</text>',
        f'<text x="72" y="70" font-family="Helvetica,Arial,sans-serif" font-size="17" fill="#666">{subtitle}</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#333" stroke-width="1.6"/>',
    ]
    min_x, max_x, min_y, max_y = embedding.bounds
    for tick_x in np.linspace(min_x, max_x, 4):
        x, _ = map_embedding_point(float(tick_x), min_y, embedding.bounds, plot_x, plot_y, plot_w, plot_h)
        parts.append(f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_h}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{plot_y + plot_h + 28}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="14" fill="#444">{tick_x:.1f}</text>')
    for tick_y in np.linspace(min_y, max_y, 4):
        _, y = map_embedding_point(min_x, float(tick_y), embedding.bounds, plot_x, plot_y, plot_w, plot_h)
        parts.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{plot_x - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="14" fill="#444">{tick_y:.1f}</text>')

    for point in embedding.public_points:
        x, y = map_embedding_point(point.x, point.y, embedding.bounds, plot_x, plot_y, plot_w, plot_h)
        color = GROUP_COLORS.get(point.group, "#999999")
        parts.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + stable_jitter(point.sample_id, 32):.1f}" y2="{y - 18 + stable_jitter(point.sample_id + "label", 16):.1f}" stroke="{color}" stroke-opacity="0.18"/>'
        )
        parts.append(
            f'<text x="{x + stable_jitter(point.sample_id, 32):.1f}" y="{y - 18 + stable_jitter(point.sample_id + "label", 16):.1f}" font-family="Helvetica,Arial,sans-serif" font-size="8" fill="{color}" opacity="0.34">{svg_escape(point.sample_id)}</text>'
        )

    for point in embedding.public_points:
        x, y = map_embedding_point(point.x, point.y, embedding.bounds, plot_x, plot_y, plot_w, plot_h)
        color = GROUP_COLORS.get(point.group, "#999999")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.2" fill="{color}" fill-opacity="0.58" stroke="#fff" stroke-width="0.6"/>')

    for point in embedding.query_points:
        x, y = map_embedding_point(point.x, point.y, embedding.bounds, plot_x, plot_y, plot_w, plot_h)
        parts.append(
            f'<polygon points="{x:.1f},{y - 20:.1f} {x - 19:.1f},{y + 16:.1f} {x + 19:.1f},{y + 16:.1f}" fill="#e1271b" stroke="#9e1610" stroke-width="1.2"/>'
        )
        parts.append(f'<text x="{x + 22:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,sans-serif" font-size="11" font-weight="700" fill="#e1271b">{svg_escape(point.sample_id)}</text>')

    parts.extend(
        [
            f'<text x="{plot_x + plot_w / 2}" y="{plot_y + plot_h + 54}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="19" font-weight="700">{"UMAP 1" if embedding.method.startswith("source_data_umap") else "Embedding 1"}</text>',
            f'<text x="24" y="{plot_y + plot_h / 2}" transform="rotate(-90 24 {plot_y + plot_h / 2})" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="19" font-weight="700">{"UMAP 2" if embedding.method.startswith("source_data_umap") else "Embedding 2"}</text>',
            f'<text x="{legend_x}" y="42" font-family="Helvetica,Arial,sans-serif" font-size="22" font-weight="700">Reference Panel</text>',
        ]
    )

    y_legend = 76
    counts = {group: sum(1 for row in public_context if row.get("group") == group) for group, _, _ in REFERENCE_GROUPS}
    for group, label, color in REFERENCE_GROUPS:
        count = counts[group]
        if count == 0:
            continue
        parts.append(f'<rect x="{legend_x}" y="{y_legend}" width="54" height="24" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 72}" y="{y_legend + 13}" font-family="Helvetica,Arial,sans-serif" font-size="17" font-weight="700">{svg_escape(group)} (n = {count})</text>')
        parts.append(f'<text x="{legend_x + 72}" y="{y_legend + 33}" font-family="Helvetica,Arial,sans-serif" font-size="15" fill="#333">{svg_escape(label)}</text>')
        y_legend += 48
    parts.append(f'<text x="72" y="536" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#666">{footer}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_dominant_distribution_svg(path: Path, public_context: list[dict], groups: list[str]) -> None:
    width = 1120
    row_h = 34
    left = 180
    right = 40
    top = 64
    height = top + row_h * len(groups) + 54
    chart_w = width - left - right
    counts = {
        group: {module: 0 for module in DISPLAY_MODULE_ORDER}
        for group in groups
    }
    totals = {group: 0 for group in groups}
    for row in public_context:
        group = row["group"]
        module = row["dominant_module"]
        if group in counts and module in counts[group]:
            counts[group][module] += 1
            totals[group] += 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Helvetica,Arial,sans-serif" font-size="20" font-weight="700">Public GEO background: dominant immune modules</text>',
        '<text x="36" y="54" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#555">GSE280220 + GSE193068, scored on the PersoMed seven-module scale</text>',
    ]
    for idx, module in enumerate(DISPLAY_MODULE_ORDER):
        x = left + idx * 112
        parts.append(f'<rect x="{x}" y="76" width="12" height="12" fill="{MODULE_COLORS[module]}"/>')
        parts.append(f'<text x="{x + 18}" y="87" font-family="Helvetica,Arial,sans-serif" font-size="11">{svg_escape(module)}</text>')

    y = top + 42
    for group in groups:
        total = totals[group]
        parts.append(f'<text x="36" y="{y + 20}" font-family="Helvetica,Arial,sans-serif" font-size="12">{svg_escape(group)} ({total})</text>')
        x = left
        for module in DISPLAY_MODULE_ORDER:
            value = counts[group][module]
            segment_w = 0 if total == 0 else chart_w * value / total
            if segment_w > 0:
                parts.append(f'<rect x="{x:.2f}" y="{y}" width="{segment_w:.2f}" height="22" fill="{MODULE_COLORS[module]}"/>')
                if segment_w >= 26:
                    parts.append(f'<text x="{x + segment_w / 2:.2f}" y="{y + 15}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="10" fill="#fff">{value}</text>')
            x += segment_w
        parts.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="22" fill="none" stroke="#333" stroke-width="0.5"/>')
        y += row_h
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def heat_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    low = (244, 247, 251)
    high = (37, 99, 153)
    rgb = tuple(round(low[i] + (high[i] - low[i]) * value) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def write_public_heatmap_svg(path: Path, public_context: list[dict], groups: list[str]) -> None:
    width = 940
    left = 176
    top = 82
    cell_w = 94
    cell_h = 32
    height = top + cell_h * len(groups) + 48
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Helvetica,Arial,sans-serif" font-size="20" font-weight="700">Public GEO background: median module activation</text>',
        '<text x="36" y="54" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#555">Values are median logistic activation scores per public cohort/group</text>',
    ]
    for idx, module in enumerate(DISPLAY_MODULE_ORDER):
        x = left + idx * cell_w
        parts.append(f'<text x="{x + cell_w / 2}" y="74" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="12" font-weight="700">{module}</text>')
    for row_idx, group in enumerate(groups):
        y = top + row_idx * cell_h
        group_rows = [row for row in public_context if row["group"] == group]
        parts.append(f'<text x="36" y="{y + 21}" font-family="Helvetica,Arial,sans-serif" font-size="12">{svg_escape(group)} ({len(group_rows)})</text>')
        for col_idx, module in enumerate(DISPLAY_MODULE_ORDER):
            values = [row[f"{module}_activation"] for row in group_rows]
            value = median(values)
            x = left + col_idx * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{heat_color(value)}" stroke="#fff"/>')
            fill = "#fff" if value > 0.62 else "#111"
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 20}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="{fill}">{value:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def stable_jitter(text: str, span: float = 28.0) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    scaled = int(digest[:8], 16) / 0xFFFFFFFF
    return (scaled - 0.5) * span


def write_input_overlay_svg(path: Path, profiles: list[SampleProfile], public_context: list[dict]) -> None:
    width = 940
    height = 420
    left = 76
    top = 62
    chart_w = 810
    chart_h = 280
    module_step = chart_w / len(DISPLAY_MODULE_ORDER)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Helvetica,Arial,sans-serif" font-size="20" font-weight="700">Input samples against public GEO background</text>',
        '<text x="36" y="54" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#555">Grey dots are public GEO samples; red diamonds are samples in this report</text>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + chart_h - tick * chart_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e4e7eb"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="10">{tick:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#333"/>')
    for idx, module in enumerate(DISPLAY_MODULE_ORDER):
        x_center = left + module_step * (idx + 0.5)
        parts.append(f'<text x="{x_center:.1f}" y="{top + chart_h + 28}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="12" font-weight="700">{module}</text>')
        for row in public_context:
            value = row[f"{module}_activation"]
            x = x_center + stable_jitter(f"{row['source_accession']}:{row['sample_id']}:{module}")
            y = top + chart_h - value * chart_h
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="#9aa7b2" opacity="0.42"/>')
        for profile in profiles:
            value = profile_activation(profile, module)
            x = x_center + stable_jitter(f"input:{profile.sample_id}:{module}", span=18)
            y = top + chart_h - value * chart_h
            parts.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" transform="rotate(45 {x:.1f} {y:.1f})" fill="#c62828" stroke="#7f1212"/>')
    parts.append('<text x="36" y="392" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#555">Activation >= 0.5 means the module is above its PersoMed threshold.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(
    profiles: list[SampleProfile],
    output_dir: Path,
    input_path: Path,
    demo: bool,
    public_context: list[dict],
    context_figures: list[dict],
    embedding: ReferenceEmbedding | None,
) -> None:
    if embedding is None:
        embedding_sentence = "No public embedding was available in this installation."
    elif embedding.method.startswith("source_data_umap"):
        embedding_sentence = (
            "The first map is a true source-data UMAP: public samples define a standardized seven-module UMAP embedding, "
            "and input samples are projected with the fitted UMAP transform. Nearest public neighbors are reported from the same score space."
        )
    else:
        embedding_sentence = (
            "The first map is source-data-derived: public samples define a standardized seven-module embedding, "
            "and input samples are placed by inverse-distance KNN projection against their nearest public references. It is not drawn from disease-label centroids."
        )
    lines = [
        "# Derm Immune Profiler Report",
        "",
        f"**Input**: `{input_path}`",
        f"**Mode**: {'demo' if demo else 'user data'}",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Method",
        "",
        "This report scores inflammatory skin expression profiles against the seven immune modules from Seremet et al. 2024 / PersoMed: Th17, Th2, Th1, type-I IFN, neutrophilic, macrophagic, and eosinophilic.",
        "Input values are expected to be log-normalized or z-scaled expression values comparable across samples. For raw NanoString RCC files, reproduce the PersoMed normalization first, then pass the module-gene matrix here.",
        "",
        "Module activation is computed as `1 / (1 + exp(-3 * (module_mean - threshold)))` using the thresholds reported in the PersoMed analysis code.",
        "",
        "## Public GEO Background",
        "",
        "The sample-level immune profile below is interpreted against a bundled public GEO context derived from GSE280220 and GSE193068. These public samples are scored with the same seven-module Derm Immune Profiler thresholds so that user data are shown against the inflammatory skin disease reference space first.",
        "",
        embedding_sentence,
        "",
    ]
    if public_context:
        lines.extend(
            [
                f"Public context samples: **{len(public_context)}** across **{len(grouped_public_context(public_context))}** cohort/group labels.",
                "",
            ]
        )
        for figure in context_figures:
            lines.extend([f"![{figure['label']}]({figure['path']})", ""])
    else:
        lines.extend(
            [
                "No public GEO context table was available in this installation, so only sample-level module scores are shown.",
                "",
            ]
        )
    lines.extend(
        [
        "## Sample Summary",
        "",
        "| Sample | Dominant module | Co-dominant modules | Interpretation |",
        "|---|---:|---|---|",
        ]
    )
    for profile in profiles:
        info = MODULE_INTERPRETATION[profile.dominant_module]
        lines.append(
            f"| {profile.sample_id} | {profile.dominant_module} | {', '.join(profile.codominant_modules)} | {info['sentinel']} |"
        )

    lines.extend(["", "## Module Scores", ""])
    for profile in profiles:
        lines.append(f"### {profile.sample_id}")
        lines.append("")
        lines.append("| Module | Raw mean | Threshold | Activation | Genes present |")
        lines.append("|---|---:|---:|---:|---:|")
        for score in profile.scores:
            raw = "NA" if math.isnan(score.raw_mean) else f"{score.raw_mean:.3f}"
            lines.append(
                f"| {score.module} | {raw} | {score.threshold:.2f} | {score.activation:.3f} | {score.genes_present}/{score.genes_expected} |"
            )
        if profile.missing_genes:
            lines.append("")
            lines.append(f"Missing signature genes: {', '.join(profile.missing_genes)}")
        lines.append("")

    lines.extend(
        [
            "## Paper Review Notes",
            "",
            "- The associated paper is Seremet et al., Nature Communications 15, 10688 (2024), DOI: 10.1038/s41467-024-54559-6.",
            "- The study uses NanoString immune-gene expression profiling of inflammatory skin biopsies to define seven functional immune modules.",
            "- The module framework is proposed for molecular cartography, diagnostic support in ambiguous cases, and treatment-target alignment, not as a standalone clinical diagnosis.",
            "- The paper explicitly notes limitations, including reduced discriminatory power for some modules such as the eosinophilic module and the need for broader future multi-omics work.",
            "",
            "## Safety",
            "",
            DISCLAIMER,
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_sample_filename(sample_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in sample_id)
    return safe.strip("._") or "sample"


def _score_map(profile: SampleProfile) -> dict[str, ModuleScore]:
    return {score.module: score for score in profile.scores}


def _draw_wrapped_text(canvas, text: str, x: float, y: float, max_width: float, line_height: float, font: str = "Helvetica", size: int = 9) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    canvas.setFont(font, size)
    words = text.split()
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if stringWidth(candidate, font, size) <= max_width:
            line = candidate
            continue
        if line:
            canvas.drawString(x, y, line)
            y -= line_height
        line = word
    if line:
        canvas.drawString(x, y, line)
        y -= line_height
    return y


def _draw_bar_chart(
    canvas,
    title: str,
    values: list[tuple[str, float]],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    y_min: float,
    y_max: float,
    threshold: float | None = None,
    y_label: str = "",
) -> None:
    from reportlab.lib import colors

    canvas.setStrokeColor(colors.black)
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawCentredString(x + width / 2, y + height + 18, title)
    canvas.setFont("Helvetica", 7)
    if y_label:
        canvas.saveState()
        canvas.translate(x - 28, y + height / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, y_label)
        canvas.restoreState()

    axis_x = x + 34
    axis_y = y + 28
    chart_w = width - 48
    chart_h = height - 44
    canvas.line(axis_x, axis_y, axis_x, axis_y + chart_h)
    canvas.line(axis_x, axis_y, axis_x + chart_w, axis_y)

    def to_y(value: float) -> float:
        clipped = max(y_min, min(y_max, value))
        return axis_y + ((clipped - y_min) / (y_max - y_min)) * chart_h

    for tick in [y_min, (y_min + y_max) / 2, y_max]:
        tick_y = to_y(tick)
        canvas.setStrokeColor(colors.lightgrey)
        canvas.line(axis_x, tick_y, axis_x + chart_w, tick_y)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(axis_x - 4, tick_y - 2, f"{tick:.1f}")
    canvas.setStrokeColor(colors.black)

    if threshold is not None:
        thresh_y = to_y(threshold)
        canvas.setDash(3, 2)
        canvas.setStrokeColor(colors.HexColor("#666666"))
        canvas.line(axis_x, thresh_y, axis_x + chart_w, thresh_y)
        canvas.setDash()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(axis_x + chart_w - 54, thresh_y + 3, f"threshold ({threshold:.1f})")

    bar_gap = 6
    bar_w = max(8, (chart_w - bar_gap * (len(values) + 1)) / len(values))
    zero_y = to_y(0) if y_min < 0 < y_max else axis_y
    for idx, (label, value) in enumerate(values):
        bar_x = axis_x + bar_gap + idx * (bar_w + bar_gap)
        value_y = to_y(value)
        top = max(value_y, zero_y)
        bottom = min(value_y, zero_y)
        color = colors.HexColor("#8fb9e8") if value < 0.5 else colors.HexColor("#2f6fb1")
        if y_min < 0:
            color = colors.HexColor("#d95f5f") if abs(value) >= 2 else colors.HexColor("#8fb9e8")
        canvas.setFillColor(color)
        canvas.rect(bar_x, bottom, bar_w, max(1, top - bottom), fill=1, stroke=0)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica", 7)
        canvas.saveState()
        canvas.translate(bar_x + bar_w / 2, axis_y - 5)
        canvas.rotate(45)
        canvas.drawRightString(0, 0, label)
        canvas.restoreState()


def _projection(profile: SampleProfile) -> tuple[float, float]:
    scores = _score_map(profile)
    x = scores["Th17"].activation + 0.5 * scores["Th1"].activation - scores["IFN"].activation
    y = scores["Th2"].activation + 0.5 * scores["Eosino"].activation - scores["Macro"].activation
    return x, y


def _draw_projection(canvas, profile: SampleProfile, all_profiles: list[SampleProfile], x: float, y: float, width: float, height: float) -> None:
    from reportlab.lib import colors

    points = [(other, *_projection(other)) for other in all_profiles]
    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    min_x, max_x = min(xs + [-1]), max(xs + [1])
    min_y, max_y = min(ys + [-1]), max(ys + [1])
    pad_x = max(0.2, (max_x - min_x) * 0.15)
    pad_y = max(0.2, (max_y - min_y) * 0.15)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(x + width / 2, y + height + 18, "Module score projection")
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(x + width / 2, y - 18, "Axis 1: Th17/Th1 vs IFN")
    canvas.saveState()
    canvas.translate(x - 28, y + height / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, "Axis 2: Th2/Eosino vs Macro")
    canvas.restoreState()

    canvas.setStrokeColor(colors.black)
    canvas.rect(x, y, width, height, fill=0, stroke=1)
    for idx in range(1, 4):
        grid_x = x + width * idx / 4
        grid_y = y + height * idx / 4
        canvas.setStrokeColor(colors.lightgrey)
        canvas.line(grid_x, y, grid_x, y + height)
        canvas.line(x, grid_y, x + width, grid_y)

    def map_x(value: float) -> float:
        return x + ((value - min_x) / (max_x - min_x)) * width

    def map_y(value: float) -> float:
        return y + ((value - min_y) / (max_y - min_y)) * height

    for other, point_x, point_y in points:
        px = map_x(point_x)
        py = map_y(point_y)
        if other.sample_id == profile.sample_id:
            canvas.setFillColor(colors.HexColor("#c62828"))
            canvas.setStrokeColor(colors.HexColor("#c62828"))
            canvas.line(px, py + 7, px - 7, py - 5)
            canvas.line(px - 7, py - 5, px + 7, py - 5)
            canvas.line(px + 7, py - 5, px, py + 7)
            canvas.setFont("Helvetica", 7)
            canvas.drawString(px + 8, py, other.sample_id[:28])
        else:
            canvas.setFillColor(colors.HexColor("#9aa7b2"))
            canvas.circle(px, py, 4, fill=1, stroke=0)


def _draw_public_context_pdf_page(
    canvas,
    page_w: float,
    page_h: float,
    public_context: list[dict],
    profiles: list[SampleProfile],
    embedding: ReferenceEmbedding | None,
) -> None:
    from reportlab.lib import colors

    groups = grouped_public_context(public_context)
    canvas.setFont("Helvetica-Bold", 16)
    canvas.setFillColor(colors.black)
    canvas.drawString(36, 750, "Public GEO Context")
    canvas.setFont("Helvetica", 9)
    canvas.drawString(36, 733, "GSE280220 and GSE193068 scored with the same PersoMed seven-module thresholds")

    is_umap = embedding is not None and embedding.method.startswith("source_data_umap")
    # Source-data embedding map
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(36, 704, "Source-data UMAP immune map" if is_umap else "Source-data KNN immune-map projection")
    plot_x, plot_y, plot_w, plot_h = 48, 456, 310, 218
    canvas.setStrokeColor(colors.HexColor("#333333"))
    canvas.rect(plot_x, plot_y, plot_w, plot_h, fill=0, stroke=1)
    if embedding is not None:
        min_x, max_x, min_y, max_y = embedding.bounds
        for tick_x in np.linspace(min_x, max_x, 4):
            x, _ = map_embedding_point(float(tick_x), min_y, embedding.bounds, plot_x, plot_y, plot_w, plot_h)
            canvas.setStrokeColor(colors.lightgrey)
            canvas.line(x, plot_y, x, plot_y + plot_h)
            canvas.setFillColor(colors.black)
            canvas.setFont("Helvetica", 6)
            canvas.drawCentredString(x, plot_y - 10, f"{tick_x:.1f}")
        for tick_y in np.linspace(min_y, max_y, 4):
            _, y = map_embedding_point(min_x, float(tick_y), embedding.bounds, plot_x, plot_y, plot_w, plot_h)
            canvas.setStrokeColor(colors.lightgrey)
            canvas.line(plot_x, y, plot_x + plot_w, y)
            canvas.setFillColor(colors.black)
            canvas.setFont("Helvetica", 6)
            canvas.drawRightString(plot_x - 5, y - 2, f"{tick_y:.1f}")
        for point in embedding.public_points:
            x, y = map_embedding_point(point.x, point.y, embedding.bounds, plot_x, plot_y, plot_w, plot_h)
            canvas.setFillColor(colors.HexColor(GROUP_COLORS.get(point.group, "#999999")))
            canvas.circle(x, y, 3.2, fill=1, stroke=0)
        for point in embedding.query_points:
            x, y = map_embedding_point(point.x, point.y, embedding.bounds, plot_x, plot_y, plot_w, plot_h)
            canvas.setFillColor(colors.HexColor("#e1271b"))
            path = canvas.beginPath()
            path.moveTo(x, y + 10)
            path.lineTo(x - 9, y - 8)
            path.lineTo(x + 9, y - 8)
            path.close()
            canvas.drawPath(path, fill=1, stroke=0)
            canvas.setFont("Helvetica", 6)
            canvas.drawString(x + 10, y, point.sample_id[:20])
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(plot_x + plot_w / 2, plot_y - 24, "UMAP 1" if is_umap else "Embedding 1")
    canvas.saveState()
    canvas.translate(plot_x - 28, plot_y + plot_h / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, "UMAP 2" if is_umap else "Embedding 2")
    canvas.restoreState()

    legend_x = 388
    legend_y = 670
    canvas.setFont("Helvetica-Bold", 11)
    canvas.setFillColor(colors.black)
    canvas.drawString(legend_x, 704, "Reference Panel")
    counts = {group: sum(1 for row in public_context if row.get("group") == group) for group, _, _ in REFERENCE_GROUPS}
    for group, label, color in REFERENCE_GROUPS[:9]:
        count = counts[group]
        if count == 0:
            continue
        canvas.setFillColor(colors.HexColor(color))
        canvas.rect(legend_x, legend_y, 24, 12, fill=1, stroke=0)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(legend_x + 32, legend_y + 4, f"{group} (n={count})")
        canvas.setFont("Helvetica", 7)
        canvas.drawString(legend_x + 96, legend_y + 4, label[:28])
        legend_y -= 20

    # Median activation heatmap
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(36, 402, "Median module activation by public group")
    heat_x = 122
    heat_y = 366
    cell_w = 58
    cell_h = 16
    for col, module in enumerate(DISPLAY_MODULE_ORDER):
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica", 7)
        canvas.drawCentredString(heat_x + col * cell_w + cell_w / 2, heat_y + 18, module)
    for row_idx, group in enumerate(groups[:12]):
        rows = [row for row in public_context if row["group"] == group]
        y0 = heat_y - row_idx * cell_h
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(heat_x - 7, y0 + 4, group[:22])
        for col, module in enumerate(DISPLAY_MODULE_ORDER):
            value = median([row[f"{module}_activation"] for row in rows])
            canvas.setFillColor(colors.HexColor(heat_color(value)))
            canvas.rect(heat_x + col * cell_w, y0, cell_w - 1, cell_h - 1, fill=1, stroke=0)
            canvas.setFillColor(colors.white if value > 0.62 else colors.black)
            canvas.drawCentredString(heat_x + col * cell_w + cell_w / 2, y0 + 4, f"{value:.2f}")

    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(36, 52, "Public context is derived from GEO GSE280220/GSE193068; query samples are projected from seven-module scores.")
    canvas.drawCentredString(page_w / 2, 36, "Page 1")
    canvas.showPage()


def _write_pdf_to_path(
    profiles: list[SampleProfile],
    pdf_path: Path,
    input_path: Path,
    demo: bool,
    public_context: list[dict],
    embedding: ReferenceEmbedding | None,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    page_w, page_h = letter
    if public_context:
        _draw_public_context_pdf_page(c, page_w, page_h, public_context, profiles, embedding)

    for page_index, profile in enumerate(profiles, start=1):
        scores = _score_map(profile)
        active = [score.module for score in profile.scores if score.activation >= 0.5]
        info = MODULE_INTERPRETATION[profile.dominant_module]
        missing_count = len(profile.missing_genes)

        c.setFont("Helvetica-Bold", 15)
        c.drawString(36, 750, f"Report for Sample: {profile.sample_id}")
        c.setFont("Helvetica", 9)
        c.drawString(36, 734, f"Input file: {input_path.name}")

        c.setFont("Helvetica-Bold", 10)
        c.drawString(36, 708, "Main Conclusions")
        c.setFont("Helvetica", 9)
        conclusion = (
            "None detected (no module exceeded the activation threshold of 0.5)."
            if not active
            else f"Dominant module: {profile.dominant_module}. Active modules: {', '.join(active)}."
        )
        y = _draw_wrapped_text(c, f"Immune axis activation: {conclusion}", 36, 692, 285, 11)
        y = _draw_wrapped_text(c, f"Interpretation: {info['sentinel']}.", 36, y - 2, 285, 11)
        _draw_wrapped_text(c, f"Co-dominant modules: {', '.join(profile.codominant_modules)}.", 36, y - 2, 285, 11)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(365, 708, "QC")
        qc_rows = [
            ("Input table", "PASS"),
            ("Module genes", "PASS" if missing_count == 0 else "WARN"),
            ("Numeric expression", "PASS"),
            ("Demo mode", "YES" if demo else "NO"),
            ("Clinical boundary", "PASS"),
        ]
        y_qc = 690
        for label, status in qc_rows:
            c.setFont("Helvetica", 9)
            c.drawString(365, y_qc, f"{label}:")
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(colors.HexColor("#2e7d32") if status in {"PASS", "YES"} else colors.HexColor("#b26a00"))
            c.drawRightString(550, y_qc, status)
            c.setFillColor(colors.black)
            y_qc -= 22

        activation_values = [(module, scores[module].activation) for module in DISPLAY_MODULE_ORDER]
        mean_values = [
            (module, 0.0 if math.isnan(scores[module].raw_mean) else scores[module].raw_mean - scores[module].threshold)
            for module in DISPLAY_MODULE_ORDER
        ]
        _draw_bar_chart(
            c,
            "Gilliet/PersoMed activation score",
            activation_values,
            42,
            392,
            245,
            145,
            y_min=0,
            y_max=1,
            threshold=0.5,
            y_label="Activation (0-1)",
        )
        _draw_bar_chart(
            c,
            "Module score vs activation threshold",
            mean_values,
            330,
            392,
            230,
            145,
            y_min=-2,
            y_max=2,
            threshold=0,
            y_label="Mean minus threshold",
        )
        _draw_projection(c, profile, profiles, 86, 88, 430, 230)

        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#444444"))
        c.drawString(36, 52, f"Processed: {generated}")
        page_offset = 1 if public_context else 0
        c.drawCentredString(page_w / 2, 52, f"Page {page_offset + 2 * page_index - 1}")
        c.drawRightString(576, 52, f"{profile.sample_id} | Derm immune profiler")
        c.showPage()

        c.setFont("Helvetica-Bold", 15)
        c.setFillColor(colors.black)
        c.drawString(36, 750, "Derm Immune Profiler Reference")
        c.setFont("Helvetica", 10)
        c.drawString(36, 730, "Seven-module inflammatory skin transcriptomic scoring report")

        c.setFont("Helvetica-Bold", 11)
        c.drawString(36, 700, "Module Reference")
        y_ref = 680
        c.setFont("Helvetica-Bold", 8)
        c.drawString(42, y_ref, "Module")
        c.drawString(100, y_ref, "Threshold")
        c.drawString(165, y_ref, "Sentinel/pathway frame")
        y_ref -= 12
        for module in MODULE_ORDER:
            c.setFont("Helvetica", 8)
            c.drawString(42, y_ref, module)
            c.drawString(100, y_ref, f"{THRESHOLDS[module]:.2f}")
            y_ref = _draw_wrapped_text(c, MODULE_INTERPRETATION[module]["sentinel"], 165, y_ref, 360, 10, size=8)
            y_ref -= 2

        c.setFont("Helvetica-Bold", 11)
        c.drawString(36, 492, "About this report")
        paragraphs = [
            "Gilliet/PersoMed Immune Module Activation Score: module means are passed through sigmoid(3*(module_mean - threshold)); a score >= 0.5 indicates activation at the published threshold.",
            "Module score vs activation threshold: the right-hand chart shows each module mean minus its activation threshold. It is a local threshold-deviation view, not a healthy-donor z-score, because this lightweight wrapper does not bundle the full sentinel reference panel.",
            "Within-run projection: this page's scatter plot compares only the samples in this report. The public-context page contains the source-data-derived reference embedding and KNN placement against GSE280220/GSE193068.",
            "Important: systemic or topical treatments before biopsy can alter transcriptomic module scores. Interpret results alongside medication history, phenotype, pathology, and clinical context.",
            DISCLAIMER,
        ]
        y_para = 472
        for paragraph in paragraphs:
            y_para = _draw_wrapped_text(c, paragraph, 36, y_para, 530, 12, size=9)
            y_para -= 10

        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#444444"))
        c.drawString(36, 52, f"Processed: {generated}")
        c.drawCentredString(page_w / 2, 52, f"Page {page_offset + 2 * page_index}")
        c.drawRightString(576, 52, f"{profile.sample_id} | Derm immune profiler")
        c.showPage()

    c.save()


def write_pdf_reports(
    profiles: list[SampleProfile],
    output_dir: Path,
    input_path: Path,
    demo: bool,
    public_context: list[dict],
    embedding: ReferenceEmbedding | None,
) -> None:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        (output_dir / "report.pdf.unavailable.txt").write_text(
            "PDF report generation requires reportlab. Install reportlab to enable report.pdf output.\n",
            encoding="utf-8",
        )
        return

    _write_pdf_to_path(profiles, output_dir / "report.pdf", input_path, demo, public_context, embedding)
    per_sample_dir = output_dir / "per_sample_reports"
    for profile in profiles:
        _write_pdf_to_path(
            [profile],
            per_sample_dir / f"{_safe_sample_filename(profile.sample_id)}_report.pdf",
            input_path,
            demo,
            public_context,
            embedding,
        )


def write_reproducibility(output_dir: Path, input_path: Path, demo: bool) -> None:
    repro_dir = output_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)
    command = "python derm_immune_profiler.py"
    if demo:
        command += " --demo"
    else:
        command += f" --input {input_path}"
    command += f" --output {output_dir}"
    (repro_dir / "commands.sh").write_text(command + "\n", encoding="utf-8")
    (repro_dir / "environment.yml").write_text(
        "name: derm-immune-profiler\nchannels:\n  - conda-forge\n  - nodefaults\ndependencies:\n  - python>=3.11\n",
        encoding="utf-8",
    )


def state_id(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def run_pipeline(input_path: Path, output_dir: Path, demo: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = score_expression_table(input_path)
    records = [profile_to_record(profile) for profile in profiles]
    public_context = load_public_context()
    embedding = build_reference_embedding(public_context, profiles)
    context_figures = write_public_context_figures(profiles, output_dir, public_context, embedding)
    write_tables(profiles, output_dir, embedding)
    write_report(profiles, output_dir, input_path, demo, public_context, context_figures, embedding)
    write_pdf_reports(profiles, output_dir, input_path, demo, public_context, embedding)
    write_reproducibility(output_dir, input_path, demo)

    result = {
        "schema": "derm_immune_profiler.result.v1",
        "source": {
            "pipeline": "derchuv/persomed",
            "paper_doi": "10.1038/s41467-024-54559-6",
            "paper_title": "Immune modules to guide diagnosis and personalized treatment of inflammatory skin diseases",
        },
        "samples": records,
        "module_order": MODULE_ORDER,
        "thresholds": THRESHOLDS,
        "public_geo_context": {
            "sources": ["GSE280220", "GSE193068"],
            "sample_count": len(public_context),
            "figures": context_figures,
            "embedding": embedding_to_record(embedding),
        },
        "workflow_state": {
            "state_schema": "derm_immune_profiler.workflow_state.v1",
            "state_id": state_id(records),
            "state_label": "immune-module-profile-ready",
            "lifecycle": "ready",
        },
        "chat_summary_lines": [
            f"Derm immune profiling complete for {len(records)} sample(s).",
            "Dominant modules: "
            + ", ".join(f"{record['sample_id']}={record['dominant_module']}" for record in records),
            "Research support only; not a clinical diagnosis or treatment prescription.",
        ],
        "preferred_artifacts": [
            {"path": "report.pdf", "label": "PDF report"},
            {"path": "report.md", "label": "Markdown report"},
            *context_figures,
            {"path": "tables/module_scores.csv", "label": "Module scores"},
            {"path": "tables/sample_summary.csv", "label": "Sample summary"},
        ],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile inflammatory skin immune modules from expression data.")
    parser.add_argument("--input", dest="input_path", type=Path, help="CSV/TSV expression table with sample_id rows and module-gene columns")
    parser.add_argument("--output", dest="output_dir", type=Path, default=Path("derm_immune_profiler_output"), help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run bundled synthetic demo")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = DEMO_EXPRESSION if args.demo else args.input_path
    if input_path is None:
        print("ERROR: provide --input or use --demo", file=sys.stderr)
        return 2
    try:
        result = run_pipeline(input_path=input_path, output_dir=args.output_dir, demo=args.demo)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("\n".join(result["chat_summary_lines"]))
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
