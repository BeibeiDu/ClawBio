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
