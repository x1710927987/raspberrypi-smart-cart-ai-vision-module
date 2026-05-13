from __future__ import annotations

import argparse
from pathlib import Path

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
