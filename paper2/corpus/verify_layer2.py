"""
Phase 2A . Layer 2 checkpoint verification.

Generates pipelines from all five mechanism-class constructors (plus the
true-negative labeler-exercise), runs each through its three-regime gate
against the REAL linker, and reports. The gate runs *as part of
generation*: a pipeline that ships is a pipeline proven to test what it
claims. Exits non-zero on any gate failure.

Run:  python -m paper2.corpus.verify_layer2     (from repo root)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from paper2.corpus.gate import gate
from paper2.corpus.mechanisms import CONSTRUCTORS, ConstructedCase
from paper2.corpus.primitives import isolated_pipeline, shadow_capture

# Two instances per constructor - proves the gate holds across seeds, not by
# luck of a single fixture.
N_PER_CLASS = 2


def _fmt_shadow(case: ConstructedCase, shadow) -> str:
    parts = []
    for o in shadow.isolated:
        parts.append(f"{o.name}={o.parent_id}@{o.confidence}" if o.fired else f"{o.name}=.")
    prod = shadow.production
    prod_s = (
        f"{prod.method}:{prod.parent_id}@{prod.confidence}"
        if prod.fired else ".(silent)"
    )
    return f"prod={prod_s} | iso[{', '.join(parts)}]"


def main() -> int:
    print("=" * 78)
    print("Phase 2A . Layer 2 checkpoint - mechanism constructors + 3-regime gate")
    print("=" * 78)

    verdicts = []
    with tempfile.TemporaryDirectory(prefix="rudriq_p2_l2_") as td:
        tmp = Path(td)
        for name, constructor in CONSTRUCTORS:
            print(f"\n[{name}]")
            for idx in range(N_PER_CLASS):
                db = tmp / f"{name}-{idx}.duckdb"
                with isolated_pipeline(db) as storage:
                    case = constructor(storage, idx)
                    shadow = shadow_capture(case.llm_input)
                    verdict = gate(case, shadow)
                verdicts.append(verdict)

                status = "PASS" if verdict.passed else "FAIL"
                tri = case.trichotomy.value
                print(f"  [{status}] {case.case_id}  "
                      f"(class={case.mechanism_class.value if case.mechanism_class else '-'}, "
                      f"trichotomy={tri}, regime={verdict.regime})")
                print(f"         {_fmt_shadow(case, shadow)}")
                if not verdict.passed:
                    for label, ok, detail in verdict.failed_checks():
                        print(f"         FAILED CHECK: {label}  [{detail}]")

    n_pass = sum(1 for v in verdicts if v.passed)
    n_total = len(verdicts)
    print("\n" + "=" * 78)
    print(f"Classes exercised: {len(CONSTRUCTORS)}  |  cases: {n_total}  |  "
          f"passed: {n_pass}  |  failed: {n_total - n_pass}")
    if n_pass != n_total:
        print("RESULT: GATE FAILURES - corpus would lie; not shippable.")
        return 1
    print("RESULT: ALL GATES GREEN - every constructor tests what it claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
