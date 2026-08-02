"""
Phase 2A · corpus-assembler checkpoint.

Proves the two commitments against real output:

  1. BYTE-IDENTICAL REPRODUCTION — assemble twice, serialize canonically,
     diff. Same config -> same bytes (the property that lets a reviewer
     re-run and verify).
  2. POOL MAPPING IN THE ARTIFACT — the serialized corpus carries the
     explicit arm->pool commitment and per-entry pool, readable without the
     harness.

Plus:
  3. SELF-PROVING — every entry's gate passed (assembly refuses otherwise).
  4. RE-MATERIALIZATION — an entry's (constructor, idx) reproduces the same
     pipeline via CONSTRUCTOR_REGISTRY, so the harness can re-run from the
     artifact alone.

Writes the artifact to paper2/corpus/artifacts/arm_a.json.
Run:  python -m paper2.corpus.verify_assembler     (from repo root)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from paper2.corpus.assembler import (
    CONSTRUCTOR_REGISTRY,
    assemble,
    to_canonical_json,
)
from paper2.corpus.primitives import isolated_pipeline, shadow_capture

_failures: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def main() -> int:
    print("=" * 78)
    print("Phase 2A . corpus-assembler checkpoint")
    print("=" * 78)

    n = 3

    # 1. Byte-identical reproduction.
    print("\nProof 1 - byte-identical reproduction (same seed/config, run twice)")
    json_a = to_canonical_json(assemble(n_per_class=n))
    json_b = to_canonical_json(assemble(n_per_class=n))
    _check("two independent assemblies produce byte-identical JSON",
           json_a == json_b,
           f"len_a={len(json_a)} len_b={len(json_b)} equal={json_a == json_b}")

    corpus_dict = json.loads(json_a)
    entries = corpus_dict["entries"]

    # 3. Self-proving: all gates green (assemble() would have raised otherwise).
    print("\nProof 3 - self-proving (every entry's gate passed at assembly)")
    all_green = all(e["gate_passed"] for e in entries)
    _check(f"all {len(entries)} entries gate_passed", all_green)

    # 2. Pool mapping in the artifact.
    print("\nProof 2 - arm->pool mapping serialized in the artifact")
    pm = corpus_dict["pool_mapping"]
    _check("header declares arm A as positive/control providing linked_true",
           pm.get("arm") == "A" and "linked_true" in pm.get("provides_pools", []))
    _check("header declares negative pools sourced from Arm C",
           pm.get("true_negative_pool_source") == "arm_c"
           and pm.get("out_of_scope_pool_source") == "arm_c")
    pools = Counter(e["pool"] for e in entries)
    _check("every Arm A entry is in the linked_true pool",
           set(pools) == {"linked_true"},
           f"pools={dict(pools)}")

    # 4. Re-materialization from the artifact.
    print("\nProof 4 - re-materialization from (constructor, idx)")
    sample = entries[0]
    builder = CONSTRUCTOR_REGISTRY[sample["constructor"]]
    import tempfile
    with tempfile.TemporaryDirectory(prefix="rudriq_remat_") as td:
        with isolated_pipeline(Path(td) / "remat.duckdb") as storage:
            case = builder(storage, sample["idx"])
            shadow = shadow_capture(case.llm_input)
    _check("re-materialized case_id matches the artifact",
           case.case_id == sample["case_id"],
           f"got {case.case_id!r} expected {sample['case_id']!r}")
    _check("re-materialized linker outcome matches the recorded expectation",
           (shadow.production.method == sample["expected_method"])
           or (not shadow.production.fired and sample["expected_method"] is None),
           f"prod_method={shadow.production.method!r} expected={sample['expected_method']!r}")

    # Write the artifact.
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "arm_a.json"
    artifact.write_text(json_a, encoding="utf-8")

    # Summary.
    mech = Counter(e["mechanism_class"] for e in entries)
    print("\n" + "-" * 78)
    print(f"corpus_version : {corpus_dict['corpus_version']}")
    print(f"entries        : {len(entries)}  ({n} per constructor x "
          f"{len(corpus_dict['config']['constructors'])} constructors)")
    print(f"mechanism mix  : {dict(mech)}")
    print(f"pool mix       : {dict(pools)}")
    print(f"artifact       : {artifact}  ({len(json_a)} bytes)")

    print("\n" + "=" * 78)
    if _failures:
        print(f"RESULT: {len(_failures)} FAILED - {', '.join(_failures)}")
        return 1
    print("RESULT: ALL PROOFS PASSED - Arm A corpus is reproducible, "
          "pool-mapped, and self-proving.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
