"""
RudriQ-specific OpenTelemetry span attribute keys.

These are added to existing OTel spans (emitted by OpenLLMetry or
AutoLineage) when the linker correlates them across domains.
"""

# The node_id of the upstream lineage operation that produced this LLM call's
# input data. Set on gen_ai.* spans when the linker matches an input to an
# AutoLineage-tracked object.
RUDRIQ_LINEAGE_PARENT = "rudriq.lineage_parent"

# Which domain produced this span. One of: "data_lineage", "llm", "linked".
RUDRIQ_DOMAIN = "rudriq.domain"

# How the linker matched the span to upstream data. One of:
# "object_identity" — id() of the input matched a tracked object
# "content_hash"    — SHA-256 of the input matched a tracked output
# "name_match"      — heuristic match by variable/column name
RUDRIQ_LINK_METHOD = "rudriq.link_method"

# Confidence of the link [0.0, 1.0]. 1.0 for object_identity, 0.8 for
# content_hash, 0.5 for name_match.
RUDRIQ_LINK_CONFIDENCE = "rudriq.link_confidence"