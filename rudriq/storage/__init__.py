"""
Local-first storage backends for RudriQ traces.

The default backend is DuckDB, which gives us:
- Embedded SQL (no separate process to manage)
- Excellent analytical query performance
- Single-file storage (easy to ship, easy to back up)
- No cloud dependencies

Other backends (PostgreSQL, ClickHouse) are available for scale, but
DuckDB is the default and what every test runs against.
"""

from rudriq.storage.duckdb_backend import DuckDBStorage, get_default_storage

__all__ = ["DuckDBStorage", "get_default_storage"]
