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

    print(f"[roomscan] loading scan: {args.scan_dir}")
    from roomscan.io.stray_scanner import load_scan
    pts = load_scan(args.scan_dir, max_frames=None)
    print(f"[roomscan] {len(pts):,} points loaded")

    print("[roomscan] extracting layout …")
    try:
        from roomscan.geometry.room_layout import extract_layout
        from roomscan.render.svg_plan import render_plan
        layout = extract_layout(pts)
        render_plan(layout, args.out / "plan.svg")
        print(f"[roomscan] floor area: {layout.floor_area_m2.value:.1f} m²")
        if layout.ceiling_height_m:
            print(f"[roomscan] ceiling height: {layout.ceiling_height_m.value:.2f} m")
    except Exception as exc:
        from roomscan.render.svg_plan import render_stub
        render_stub(args.out / "plan.svg", f"Layout error: {exc}")
        print(f"[roomscan] layout failed: {exc}")

    print(f"[roomscan] → {args.out}/plan.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
