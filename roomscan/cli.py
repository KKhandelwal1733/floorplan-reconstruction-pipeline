"""CLI entry point — stubs through phase 1; each phase wires in its stage."""
import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout: a default Windows console (cp1252) otherwise raises
# UnicodeEncodeError on the first non-ASCII character and crashes the run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

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

    print("[roomscan] extracting layout...")
    try:
        from roomscan.geometry.openings import detect_openings
        from roomscan.geometry.room_layout import extract_layout
        from roomscan.render.svg_plan import render_plan
        layout = extract_layout(pts)
        render_plan(layout, args.out / "plan.svg")
        print(f"[roomscan] floor area: {layout.floor_area_m2.value:.1f} sq m")
        if layout.ceiling_height_m:
            tag = " (unobserved - wide estimate)" if layout.ceiling_unobserved else ""
            print(f"[roomscan] ceiling height: {layout.ceiling_height_m.value:.2f} m{tag}")
        for w in layout.capture_warnings:
            print(f"[roomscan] warning: {w}")

        openings = detect_openings(pts, layout)
        print(f"[roomscan] {len(openings)} opening(s) detected")
        for o in openings:
            print(f"[roomscan]   {o.kind} on wall {o.wall_idx}: "
                  f"{o.width_m.value:.2f} m wide x {o.height_m.value:.2f} m tall, "
                  f"sill {o.sill_height_m:.2f} m")
    except Exception as exc:
        from roomscan.render.svg_plan import render_stub
        render_stub(args.out / "plan.svg", f"Layout error: {exc}")
        print(f"[roomscan] layout failed: {exc}")

    print(f"[roomscan] -> {args.out}/plan.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
