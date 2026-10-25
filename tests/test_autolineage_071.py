"""RudriQ on AutoLineage 0.7.1: real pandas records, no self-loops, right node kinds.

AutoLineage 0.7.0 recorded child_id == parent_id for most derived pandas
frames (pandas copies DataFrame.attrs). 0.7.1 gives every derived frame its
own lineage id. These tests drive real pandas through the hooks and check
what lands in RudriQ's storage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

autolineage = pytest.importorskip("autolineage")
pd = pytest.importorskip("pandas")

from autolineage.core import TransformationRecord  # noqa: E402
from autolineage.core.tracker import UnifiedTracker  # noqa: E402
from autolineage.hooks.registry import HookRegistry  # noqa: E402

import rudriq.auto as _rudriq_auto  # noqa: E402
from rudriq.core.schema import EdgeKind, NodeKind  # noqa: E402
from rudriq.linker import clear_object_registry  # noqa: E402


def _version_tuple(v: str):
    return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())


pytestmark = pytest.mark.skipif(
    _version_tuple(autolineage.__version__) < (0, 7, 1),
    reason="needs autolineage>=0.7.1 (distinct lineage ids on derived frames)",
)


@pytest.fixture
def storage(tmp_path: Path, monkeypatch):
    from rudriq.storage import duckdb_backend

    db = duckdb_backend.DuckDBStorage(db_path=tmp_path / "al071.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)
    monkeypatch.setattr(_rudriq_auto, "_autolineage_run_id", {"value": None})
    clear_object_registry()
    yield db
    db.close()


def test_real_pandas_chain_has_no_self_loop_edges(storage):
    _rudriq_auto._set_autolineage_run_id("al071-chain")
    tracker = UnifiedTracker()
    tracker.register_post_record_callback(_rudriq_auto._make_mirror_callback(tracker))
    registry = HookRegistry()
    registry.install_all(tracker)
    try:
        df = pd.DataFrame({"a": [3, 1, 2, 5], "b": ["x", "y", "x", "z"]})
        out = df[df["a"] > 1].sort_values("a").rename(columns={"b": "c"}).reset_index(drop=True)
        assert len(out) == 3
    finally:
        registry.uninstall_all()

    graph = storage.load_run("al071-chain")
    assert graph is not None and len(graph.nodes) >= 3
    direct = [e for e in graph.edges if e.kind == EdgeKind.DIRECT]
    assert direct, "expected parent->child edges from the mirrored records"
    assert all(e.parent_id != e.child_id for e in direct)


@pytest.mark.parametrize("category,expected", [
    ("train", NodeKind.MODEL_TRAIN),
    ("predict", NodeKind.MODEL_PREDICT),
    ("evaluate", NodeKind.METRIC_EVAL),
    ("preprocess", NodeKind.DATA_TRANSFORM),
    ("split", NodeKind.DATA_TRANSFORM),
    ("transform", NodeKind.DATA_TRANSFORM),
    ("model_train", NodeKind.MODEL_TRAIN),  # older spelling still accepted
])
def test_autolineage_categories_map_to_node_kinds(category, expected):
    rec = TransformationRecord(category=category, operation="op")
    assert _rudriq_auto._autolineage_record_to_node_kind(rec) == expected


def test_record_without_child_id_is_skipped(storage):
    """Configure and score records carry no child id; mirroring skips them."""
    _rudriq_auto._set_autolineage_run_id("al071-configure")
    cb = _rudriq_auto._make_mirror_callback(UnifiedTracker())
    cb(TransformationRecord(category="configure", operation="OneHotEncoder.__init__"))
    graph = storage.load_run("al071-configure")
    assert graph is None or graph.nodes == []
