from __future__ import annotations

import json
import sys
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_DIR))

from derm_immune_profiler import (  # noqa: E402
    DEMO_EXPRESSION,
    MODULE_ORDER,
    load_signature,
    run_pipeline,
    score_expression_table,
)


def test_signature_contains_expected_modules():
    modules = load_signature()
    assert list(modules) == MODULE_ORDER
    assert len(modules["Th17"]) == 21
    assert len(modules["IFN"]) == 22
    assert len(modules["Th2"]) == 4


def test_demo_scoring_detects_expected_dominant_modules():
    profiles = score_expression_table(DEMO_EXPRESSION)
    observed = {profile.sample_id: profile.dominant_module for profile in profiles}
    assert observed["demo_pso_like"] == "Th17"
    assert observed["demo_ad_like"] == "Th2"
    assert observed["demo_ifn_like"] == "IFN"


def test_demo_pipeline_writes_contract_outputs(tmp_path):
    result = run_pipeline(DEMO_EXPRESSION, tmp_path, demo=True)
    assert result["schema"] == "derm_immune_profiler.result.v1"
    assert result["source"]["paper_doi"] == "10.1038/s41467-024-54559-6"
    assert len(result["samples"]) == 3
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.pdf").exists()
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "tables" / "module_scores.csv").exists()
    assert (tmp_path / "tables" / "sample_summary.csv").exists()
    assert (tmp_path / "reproducibility" / "commands.sh").exists()
    assert (tmp_path / "per_sample_reports" / "demo_pso_like_report.pdf").exists()

    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["workflow_state"]["lifecycle"] == "ready"
    assert saved["workflow_state"]["state_id"].startswith("sha256:")
    report = (tmp_path / "report.md").read_text()
    assert "research" in report
    assert "not a medical device" in report
