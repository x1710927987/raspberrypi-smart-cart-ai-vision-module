from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.service import SmartCartService


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the smart cart deployment service.")
    parser.add_argument(
        "--config",
        default=Path(__file__).resolve().parent / "config.yaml",
        type=Path,
        help="Path to the deployment YAML config.",
    )
    args = parser.parse_args()

    service = SmartCartService(args.config)
    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
