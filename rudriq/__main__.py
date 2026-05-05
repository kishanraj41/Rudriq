"""Allow `python -m rudriq <command>` invocation."""

from rudriq.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
