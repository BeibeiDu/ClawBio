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
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SIGNATURE_PATH = SCRIPT_DIR / "data" / "der_signature.csv"
DEMO_EXPRESSION = SCRIPT_DIR / "example_data" / "demo_expression.csv"

MODULE_ORDER = ["Th1", "Th2", "Th17", "Neutro", "Macro", "Eosino", "IFN"]
DISPLAY_MODULE_ORDER = ["Th2", "Th17", "Neutro", "Th1", "Eosino", "IFN", "Macro"]
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


def write_tables(profiles: list[SampleProfile], output_dir: Path) -> None:
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


def write_report(profiles: list[SampleProfile], output_dir: Path, input_path: Path, demo: bool) -> None:
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
        "## Sample Summary",
        "",
        "| Sample | Dominant module | Co-dominant modules | Interpretation |",
        "|---|---:|---|---|",
    ]
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


def _write_pdf_to_path(profiles: list[SampleProfile], pdf_path: Path, input_path: Path, demo: bool) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    page_w, page_h = letter

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
        c.drawCentredString(page_w / 2, 52, f"Page {2 * page_index - 1}")
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
            "Module score projection: the scatter plot is a deterministic projection of the samples in this run based on module activations. It is designed for quick visual comparison and should not be interpreted as the fixed PersoMed sentinel-panel UMAP.",
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
        c.drawCentredString(page_w / 2, 52, f"Page {2 * page_index}")
        c.drawRightString(576, 52, f"{profile.sample_id} | Derm immune profiler")
        c.showPage()

    c.save()


def write_pdf_reports(profiles: list[SampleProfile], output_dir: Path, input_path: Path, demo: bool) -> None:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        (output_dir / "report.pdf.unavailable.txt").write_text(
            "PDF report generation requires reportlab. Install reportlab to enable report.pdf output.\n",
            encoding="utf-8",
        )
        return

    _write_pdf_to_path(profiles, output_dir / "report.pdf", input_path, demo)
    per_sample_dir = output_dir / "per_sample_reports"
    for profile in profiles:
        _write_pdf_to_path([profile], per_sample_dir / f"{_safe_sample_filename(profile.sample_id)}_report.pdf", input_path, demo)


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
    write_tables(profiles, output_dir)
    write_report(profiles, output_dir, input_path, demo)
    write_pdf_reports(profiles, output_dir, input_path, demo)
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
