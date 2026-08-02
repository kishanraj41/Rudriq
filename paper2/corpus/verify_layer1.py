"""
Phase 2A . Layer 1 checkpoint verification.

Proves the primitive is correct against the *real* linker before any
mechanism constructor is built on top. Six proofs:

  A. object_identity round-trip (identity surface, conf 1.0)
  B. content_hash round-trip   (storage surface, conf 0.8)
  C. substring round-trip      (identity/content surface, conf 0.7/0.5)
  D. cross-pipeline STORAGE isolation  (no phantom content_hash hits)
  E. cross-pipeline REGISTRY isolation (no phantom identity hits)
  F. shadow capture surfaces a MASKED strategy in a real collision
     (object_identity vs substring -> different parents; production picks
     the higher-precedence one) - de-risks the layer 3/4 coupling.

Run:  python -m paper2.corpus.verify_layer1     (from repo root)
Exits non-zero on any failed proof.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rudriq.linker import register_object_identity

from paper2.corpus.primitives import (
    ShadowResult,
    isolated_pipeline,
    register_identity_surface,
    register_storage_surface,
    shadow_capture,
)

# Seeds - all >= 20 chars so substring thresholds (min_match_length=20) are
# satisfiable where intended.
DOC_A = "Annual revenue grew 14 percent across the EMEA region in fiscal 2025."
DOC_B = "Patient cohort 7 reported elevated biomarker readings during week three."

_failures: list[str] = []


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f"  -  {detail}"
    print(line)
    if not condition:
        _failures.append(label)


def _fmt(result: ShadowResult) -> str:
    parts = []
    for o in result.isolated:
        parts.append(f"{o.name}={o.parent_id}@{o.confidence}" if o.fired else f"{o.name}=.")
    prod = result.production
    prod_s = f"{prod.method}:{prod.parent_id}@{prod.confidence}" if prod.fired else ".(silent)"
    return f"production={prod_s} | isolated[{', '.join(parts)}]"


def proof_a(tmp: Path) -> None:
    print("\nProof A - object_identity round-trip (identity surface)")
    with isolated_pipeline(tmp / "a.duckdb"):
        register_identity_surface(DOC_A, "node-A")
        r = shadow_capture(DOC_A)
        print(f"    {_fmt(r)}")
        oi = r.isolated_by_name("object_identity")
        _check("object_identity fires to node-A at 1.0",
               oi.fired and oi.parent_id == "node-A" and oi.confidence == 1.0)
        _check("production picks object_identity",
               r.production.method == "object_identity" and r.production.parent_id == "node-A")


def proof_b(tmp: Path) -> None:
    print("\nProof B - content_hash round-trip (storage surface ONLY)")
    with isolated_pipeline(tmp / "b.duckdb") as storage:
        stored = register_storage_surface(DOC_A, "node-H", storage)
        # Exact-copy input: same content, never identity-registered.
        llm_input = "".join(list(DOC_A))  # distinct object, identical content
        r = shadow_capture(llm_input)
        print(f"    stored_hash={stored[:12]}...")
        print(f"    {_fmt(r)}")
        ch = r.isolated_by_name("content_hash")
        oi = r.isolated_by_name("object_identity")
        sub = r.isolated_by_name("substring")
        _check("content_hash fires to node-H at 0.8",
               ch.fired and ch.parent_id == "node-H" and ch.confidence == 0.8)
        _check("object_identity does NOT fire (storage surface only)", not oi.fired)
        _check("substring does NOT fire (content registry empty)", not sub.fired)
        _check("production picks content_hash",
               r.production.method == "content_hash" and r.production.parent_id == "node-H")


def proof_c(tmp: Path) -> None:
    print("\nProof C - substring round-trip (identity/content surface)")
    with isolated_pipeline(tmp / "c.duckdb"):
        register_identity_surface(DOC_B, "node-S")
        messages = [
            {"role": "system", "content": "Answer based on the provided context."},
            {"role": "user", "content": f"Context:\n{DOC_B}\n\nQuestion: summarise."},
        ]
        r = shadow_capture(messages)
        print(f"    {_fmt(r)}")
        sub = r.isolated_by_name("substring")
        oi = r.isolated_by_name("object_identity")
        _check("substring fires to node-S",
               sub.fired and sub.parent_id == "node-S")
        _check("substring confidence in {0.5, 0.7}", sub.confidence in (0.5, 0.7))
        _check("object_identity does NOT fire (fresh messages list)", not oi.fired)
        _check("production picks substring",
               r.production.method == "substring" and r.production.parent_id == "node-S")


def proof_d(tmp: Path) -> None:
    print("\nProof D - cross-pipeline STORAGE isolation")
    # Pipeline 1 persists a content_hash node.
    with isolated_pipeline(tmp / "d1.duckdb") as s1:
        register_storage_surface(DOC_A, "leak-H", s1)
        r1 = shadow_capture(DOC_A)
        _check("pipeline-1 content_hash fires", r1.isolated_by_name("content_hash").fired)
    # Pipeline 2: fresh isolated storage - must NOT see pipeline-1's node.
    with isolated_pipeline(tmp / "d2.duckdb"):
        r2 = shadow_capture(DOC_A)
        print(f"    pipeline-2 {_fmt(r2)}")
        _check("pipeline-2 content_hash does NOT fire (no storage leakage)",
               not r2.isolated_by_name("content_hash").fired)


def proof_e(tmp: Path) -> None:
    print("\nProof E - cross-pipeline REGISTRY isolation")
    with isolated_pipeline(tmp / "e1.duckdb"):
        register_identity_surface(DOC_A, "leak-A")
        r1 = shadow_capture(DOC_A)
        _check("pipeline-1 object_identity fires", r1.isolated_by_name("object_identity").fired)
    with isolated_pipeline(tmp / "e2.duckdb"):
        r2 = shadow_capture(DOC_A)
        print(f"    pipeline-2 {_fmt(r2)}")
        _check("pipeline-2 object_identity does NOT fire (no registry leakage)",
               not r2.isolated_by_name("object_identity").fired)


def proof_f(tmp: Path) -> None:
    print("\nProof F - shadow surfaces a MASKED strategy (collision preview)")
    with isolated_pipeline(tmp / "f.duckdb"):
        # Identity-only parent P_A (NOT in the content registry, so substring
        # cannot self-match it).
        register_object_identity(DOC_A, "P_A", extract_content=False)
        # Substring parent P_B (in the content registry).
        register_identity_surface(DOC_B, "P_B")
        # One input satisfying BOTH strategies at DIFFERENT parents: a list
        # whose element is the identity-registered DOC_A, and which also
        # embeds DOC_B's text as a string.
        llm_input = [DOC_A, f"Reference material: {DOC_B} - end of reference."]
        r = shadow_capture(llm_input)
        print(f"    {_fmt(r)}")
        oi = r.isolated_by_name("object_identity")
        sub = r.isolated_by_name("substring")
        _check("object_identity fires in isolation -> P_A",
               oi.fired and oi.parent_id == "P_A")
        _check("substring fires in isolation -> P_B",
               sub.fired and sub.parent_id == "P_B")
        _check("the two strategies point at DIFFERENT parents (real collision)",
               oi.parent_id != sub.parent_id)
        _check("production picks the higher-precedence strategy (object_identity)",
               r.production.method == "object_identity" and r.production.parent_id == "P_A")
        _check("substring's parent P_B is MASKED in production but visible in shadow",
               r.production.parent_id == "P_A" and sub.parent_id == "P_B")


def main() -> int:
    print("=" * 72)
    print("Phase 2A . Layer 1 checkpoint - primitives verified against real linker")
    print("=" * 72)
    with tempfile.TemporaryDirectory(prefix="rudriq_p2_l1_") as td:
        tmp = Path(td)
        proof_a(tmp)
        proof_b(tmp)
        proof_c(tmp)
        proof_d(tmp)
        proof_e(tmp)
        proof_f(tmp)

    print("\n" + "=" * 72)
    if _failures:
        print(f"RESULT: {len(_failures)} FAILED - {', '.join(_failures)}")
        return 1
    print("RESULT: ALL PROOFS PASSED - layer 1 primitive is correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
