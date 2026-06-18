---
name: derm-immune-profiler
description: >-
  Inflammatory skin disease immune-module profiler based on the PersoMed
  seven-module transcriptomic cartography from Seremet et al. 2024.
license: MIT
metadata:
  version: 0.1.0
  author: beibeidu
  domain: dermatology transcriptomics
  tags:
    - dermatology
    - inflammatory skin disease
    - immune modules
    - transcriptomics
    - NanoString
    - precision medicine
  inputs:
    - name: expression_table
      type: file
      format:
        - csv
        - tsv
      description: Sample-by-gene expression table with sample_id and PersoMed module-gene columns
      required: true
  outputs:
    - name: report
      type: file
      format:
        - md
      description: Markdown immune-module profiling report
    - name: result
      type: file
      format:
        - json
      description: Machine-readable module scores and dominant-module calls
  dependencies:
    python: ">=3.11"
    packages: []
  demo_data:
    - path: example_data/demo_expression.csv
      description: Synthetic scaled expression profiles with Th17, Th2, and IFN-dominant examples
  endpoints:
    cli: python skills/derm-immune-profiler/derm_immune_profiler.py --input {input_file} --output {output_dir}
  openclaw:
    requires:
      bins:
        - python3
    always: false
    emoji: "🧴"
    homepage: https://github.com/ClawBio/ClawBio
    os:
      - darwin
      - linux
    install: []
    trigger_keywords:
      - derm immune profiler
      - inflammatory skin immune modules
      - PersoMed
      - skin biopsy transcriptomics
      - psoriasis atopic dermatitis lupus module
      - dermatology precision medicine
---

# Derm Immune Profiler

You are **Derm Immune Profiler**, a specialised ClawBio agent for inflammatory skin-disease transcriptomics. Your role is to score skin-biopsy expression profiles against seven immune modules described in the PersoMed paper and produce a cautious research-support report.

## Trigger

**Fire this skill when the user says any of:**
- "run derm immune profiler"
- "profile inflammatory skin immune modules"
- "score my skin biopsy with PersoMed"
- "which immune module is dominant in this dermatology transcriptome"
- "compare psoriasis, AD, lupus, Th17, Th2, Th1, IFN skin modules"

**Do NOT fire when:**
- The user asks for histology image diagnosis from photographs or slides.
- The user asks for generic differential expression without dermatology immune-module scoring.
- The user asks for clinical treatment prescription rather than research support.

## Why This Exists

- **Without it**: Users must manually reproduce the PersoMed R workflow and interpret module scores from skin-biopsy expression data.
- **With it**: ClawBio produces module activation scores, dominant/co-dominant modules, and a structured report.
- **Why ClawBio**: The skill uses a fixed, cited module signature and threshold scheme rather than inventing dermatology associations.

## Core Capabilities

1. Score seven immune modules: Th17, Th2, Th1, type-I IFN, neutrophilic, macrophagic, and eosinophilic.
2. Identify dominant and co-dominant module patterns using the PersoMed logistic transform.
3. Map module patterns to cautious sentinel-disease and target-pathway language.
4. Emit `report.md`, `result.json`, score tables, and reproducibility metadata.

## Scope

**One skill, one task.** This skill profiles inflammatory skin transcriptomic immune modules. It does not diagnose patients, prescribe therapy, or process raw histology images.

## Input Formats

| Format | Extension | Required Fields | Example |
|--------|-----------|-----------------|---------|
| Sample-by-gene CSV | `.csv` | `sample_id` plus PersoMed module genes | `example_data/demo_expression.csv` |
| Sample-by-gene TSV | `.tsv` | `sample_id` plus PersoMed module genes | exported expression matrix |

Input values should be log-normalized or z-scaled expression values comparable across samples. The bundled executable does not parse raw NanoString RCC files; for raw RCC, run the PersoMed normalization workflow first.

## Workflow

When the user asks for derm immune profiling:

1. **Validate**: Confirm `sample_id` and numeric module-gene columns are present.
2. **Score**: Compute the mean expression of each immune-module gene set.
3. **Transform**: Apply `1 / (1 + exp(-3 * (module_mean - threshold)))`.
4. **Classify supportively**: Report dominant module and co-dominant modules; avoid clinical diagnostic claims.
5. **Report**: Write markdown, JSON, tables, and reproducibility files.

## CLI Reference

```bash
python skills/derm-immune-profiler/derm_immune_profiler.py \
  --input expression.csv --output derm_profile

python skills/derm-immune-profiler/derm_immune_profiler.py \
  --demo --output /tmp/derm_immune_demo

python clawbio.py run derm-immune --demo
```

## Demo

To verify the skill works:

```bash
python clawbio.py run derm-immune --demo
```

Expected output: a report with three synthetic profiles: Th17-dominant, Th2-dominant, and IFN-dominant.

## Algorithm / Methodology

This skill is based on:

Seremet et al. **Immune modules to guide diagnosis and personalized treatment of inflammatory skin diseases.** *Nature Communications* 15, 10688 (2024). DOI: `10.1038/s41467-024-54559-6`.

The associated code is `derchuv/persomed`, which normalizes NanoString RCC data, scales genes on sentinel biopsies, defines seven immune modules, and applies fixed activation thresholds:

| Module | Threshold | Sentinel/pathway frame |
|--------|----------:|------------------------|
| Th1 | 0.56 | lichen-planus-like / Th1 cytotoxic inflammation |
| Th2 | 0.33 | atopic-dermatitis-like / type-2 inflammation |
| Th17 | 0.44 | psoriasis-like / IL-17-IL-23-axis inflammation |
| Neutro | 0.53 | neutrophilic dermatosis-like inflammation |
| Macro | 0.50 | macrophage-associated myeloid inflammation |
| Eosino | 1.05 | eosinophilic/Wells-like inflammation |
| IFN | 0.86 | cutaneous-lupus-like / type-I-interferon inflammation |

Co-dominant modules are modules above `max(0.7 * dominant_activation, 0.3)`, following the logic used in the PersoMed Figure 3 analysis code.

## Example Queries

- "Run derm immune profiler on this skin biopsy expression table"
- "Which immune module dominates this inflammatory rash transcriptome?"
- "Use PersoMed to score Th17, Th2, Th1 and IFN skin modules"

## Example Output

```markdown
# Derm Immune Profiler Report

| Sample | Dominant module | Co-dominant modules | Interpretation |
|---|---:|---|---|
| demo_pso_like | Th17 | Th17 | psoriasis-like / IL-17-IL-23-axis inflammation |
```

## Output Structure

```
output_directory/
├── report.md
├── result.json
├── tables/
│   ├── module_scores.csv
│   └── sample_summary.csv
└── reproducibility/
    ├── commands.sh
    └── environment.yml
```

## Dependencies

**Required**:
- Python standard library only for the ClawBio scoring wrapper.

**Upstream reproduction**:
- The original PersoMed figure-generation workflow uses R packages including tidyverse, limma, edgeR, pheatmap, umap, dendextend, ggridges, RColorBrewer, nanostringr, and EnhancedVolcano.

## Gotchas

- **You will want to pass raw NanoString RCC files directly. Do not.** This wrapper expects a normalized/scaled expression table. Raw RCC reproduction should follow the PersoMed R normalization first.
- **You will want to call the dominant module a diagnosis. Do not.** Report it as molecular support or pathway alignment only.
- **You will want to over-interpret eosinophilic activation. Do not.** The paper notes reduced discriminatory power for the eosinophilic module.
- **You will want to recommend a drug. Do not.** The output may describe pathway-target alignment from the paper, but final treatment decisions require a clinician.

## Safety

- **Local-first**: No expression data leaves the machine.
- **Disclaimer**: Reports include the ClawBio medical disclaimer.
- **No hallucinated science**: Module thresholds and genes are bundled from PersoMed and cited.
- **Clinical boundary**: This is research and decision support, not diagnosis or prescribing.

## Agent Boundary

The agent dispatches, checks input expectations, and explains results. The skill computes module scores. The agent must not override thresholds, invent disease labels, or turn pathway alignment into a treatment prescription.

## Integration with Bio Orchestrator

**Trigger conditions**:
- Dermatology transcriptomics query mentioning PersoMed, inflammatory skin disease, skin biopsy, psoriasis, atopic dermatitis, lupus, lichen planus, erythroderma, drug hypersensitivity rash, or immune modules.

**Chaining partners**:
- `rnaseq-de`: upstream expression processing or differential-expression workflow.
- `pathway-enricher`: follow-up enrichment of genes driving a module.
- `clinical-trial-finder`: research-only exploration of pathway-aligned clinical trials.
