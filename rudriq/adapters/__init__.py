"""
Format adapters: convert between RudriQ canonical schema and external
formats (OTel, JSON files, future formats).

The canonical schema (rudriq.core.schema) is the source of truth.
Adapters translate to/from it; they never bypass it.
"""

from rudriq.adapters.otel_ingest import otel_span_to_node, classify_span_kind

__all__ = ["otel_span_to_node", "classify_span_kind"]
