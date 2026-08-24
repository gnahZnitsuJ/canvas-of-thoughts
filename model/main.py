"""Compatibility entrypoint for the historical model CLI command."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from launcher.cli import legacy_main

if __name__ == "__main__":
    raise SystemExit(legacy_main())
