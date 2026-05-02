"""
RudriQ — Connect LLM behavior to its upstream data lineage.

When your AI system breaks, RudriQ walks back through your data pipeline
to the operation that caused it.

Architecture
------------
RudriQ is a *bridge*, not a tracing library. It builds on top of:

* OpenLLMetry (Traceloop) for LLM call instrumentation, emitted as
  OpenTelemetry GenAI semantic conventions.
* AutoLineage for data pipeline instrumentation (pandas, scikit-learn,
  PySpark).

The novelty is the *linker* (rudriq.linker) which correlates spans from
both domains into a single unified trace, enabling cross-domain root-cause
analysis when LLM behavior changes.
"""

__version__ = "0.0.2.dev0"

from rudriq.core.tracker import get_tracker
from rudriq.analyzer.diagnose import diagnose

__all__ = ["get_tracker", "diagnose", "__version__"]