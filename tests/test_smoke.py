"""Smoke tests verifying package imports and public API."""

from __future__ import annotations


def test_package_imports() -> None:
    import rudriq

    assert rudriq.__version__ == "0.0.2.dev0"
    assert hasattr(rudriq, "get_tracker")
    assert hasattr(rudriq, "diagnose")


def test_tracker_is_singleton() -> None:
    from rudriq import get_tracker

    a = get_tracker()
    b = get_tracker()
    assert a is b


def test_tracker_starts_empty() -> None:
    from rudriq import get_tracker

    tracker = get_tracker()
    graph = tracker.get_full_graph()
    assert graph["node_count"] == 0
    assert graph["version"] == "0.0.1"


def test_diagnose_returns_structural_summary_when_empty() -> None:
    from rudriq import diagnose

    result = diagnose()
    assert result["summary"]["node_count"] == 0
    assert result["root_causes"] == []
    assert "v0.0.1" in result["notes"][0]


def test_diagnose_accepts_target_and_baseline() -> None:
    from rudriq import diagnose

    result = diagnose(target_metric="answer_quality", baseline_path="/tmp/baseline.json")
    assert result["summary"]["target_metric"] == "answer_quality"
    assert result["summary"]["has_baseline"] is True


def test_audit_report_returns_default_structure() -> None:
    from rudriq.export.audit import generate_audit_report

    result = generate_audit_report()
    assert result["report_version"] == "0.0.1"
    assert "data_operations" in result
    assert "llm_operations" in result
    assert "cross_domain_links" in result


def test_audit_unsupported_template_raises() -> None:
    import pytest

    from rudriq.export.audit import generate_audit_report

    with pytest.raises(NotImplementedError):
        generate_audit_report(template="custom-internal")


def test_auto_import_does_not_explode() -> None:
    """`import rudriq.auto` runs cleanly even with no extras installed."""
    import rudriq.auto  # noqa: F401