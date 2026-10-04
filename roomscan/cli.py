"""CLI entry point — stubs through phase 1; each phase wires in its stage."""
import argparse
import sys
from pathlib import Path

from roomscan.render.svg_plan import render_stub


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="roomscan", description="Floor plan reconstruction pipeline")
    p.add_argument("scan_dir", type=Path, help="Directory containing scan data")
    p.add_argument("--out", type=Path, default=Path("out"), help="Output directory (default: out/)")
    p.add_argument("--tier", choices=["lidar", "video", "photo"], default=None,
                   help="Force capture tier (default: auto-detect)")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # ponytail: stub — each phase replaces this with its stage call
    render_stub(args.out / "plan.svg", f"roomscan stub — scan_dir={args.scan_dir}")
    print(f"[roomscan] stub run complete → {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
