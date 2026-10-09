"""Methodology validation with known ground truth (slow; run with -m slow).

Positive control: Lighter lags the reference venue by a planted delay, so a
real, costed edge exists. Negative control: no lag. The pipeline must find a
credible edge in the first and must NOT in the second.
"""

import json

import pytest

from apexmind.cli import main


@pytest.mark.slow
def test_positive_and_negative_controls(tmp_path):
    code = main(["validate-methodology", "--out", str(tmp_path), "--hours", "6", "--lag", "3", "--seed", "5"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["positive_control"]["selected"] is not None, summary
    assert summary["negative_control"]["selected"] is None, summary
    assert code == 0
    for tag in ("positive_control", "negative_control"):
        report = (tmp_path / tag / "run" / "report.md").read_text()
        assert "SYNTHETIC DATA" in report and "LATENCY NOT MEASURED" in report
