"""CLI entry point — each phase wires in its stage."""
import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout: a default Windows console (cp1252) otherwise raises
# UnicodeEncodeError on the first non-ASCII character and crashes the run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}


def _detect_tier(input_path: Path) -> str:
    """Auto-detect capture tier from the input path shape."""
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    if input_path.is_dir() and (input_path / "depth").exists():
        return "lidar"
    if input_path.is_dir():
        return "photo"
    raise ValueError(f"Could not detect capture tier for {input_path}")


def _run_lidar(input_path: Path, out: Path) -> None:
    print(f"[roomscan] loading scan: {input_path}")
    from roomscan.io.stray_scanner import load_scan
    pts = load_scan(input_path, max_frames=None)
    print(f"[roomscan] {len(pts):,} points loaded")

    print("[roomscan] extracting layout...")
    try:
        from roomscan.geometry.openings import detect_openings
        from roomscan.geometry.room_layout import extract_layout
        from roomscan.render.svg_plan import render_plan
        layout = extract_layout(pts)
        render_plan(layout, out / "plan.svg")
        _print_layout_summary(layout)

        openings = detect_openings(pts, layout)
        print(f"[roomscan] {len(openings)} opening(s) detected")
        for o in openings:
            print(f"[roomscan]   {o.kind} on wall {o.wall_idx}: "
                  f"{o.width_m.value:.2f} m wide x {o.height_m.value:.2f} m tall, "
                  f"sill {o.sill_height_m:.2f} m")
    except Exception as exc:
        from roomscan.render.svg_plan import render_stub
        render_stub(out / "plan.svg", f"Layout error: {exc}")
        print(f"[roomscan] layout failed: {exc}")


def _run_video(input_path: Path, out: Path) -> None:
    print(f"[roomscan] loading video: {input_path}")
    try:
        from roomscan.geometry.video_tier import process_video
        from roomscan.render.svg_plan import render_plan
        layout, diagnostics = process_video(input_path)
        render_plan(layout, out / "plan.svg")
        _print_layout_summary(layout)
        print(f"[roomscan] video tier: {diagnostics['n_frames_sampled']} frames sampled, "
              f"{diagnostics['n_points_unscaled']} sparse points reconstructed, "
              f"scale +/-{diagnostics['scale_rel_half_width']:.0%}")
    except Exception as exc:
        from roomscan.render.svg_plan import render_stub
        render_stub(out / "plan.svg", f"Video reconstruction error: {exc}")
        print(f"[roomscan] video tier failed: {exc}")


def _run_photo(input_path: Path, out: Path) -> None:
    print(f"[roomscan] loading property folder: {input_path}")
    try:
        from roomscan.geometry.photo_tier import process_property
        from roomscan.render.svg_plan import render_property
        property_layout, diagnostics = process_property(input_path)
        render_property(property_layout, out / "plan.svg")
        print(f"[roomscan] {diagnostics['n_rooms_reconstructed']}/{diagnostics['n_rooms_found']} "
              f"room(s) reconstructed")
        for room in property_layout.rooms:
            print(f"[roomscan]   room '{room.name}': "
                  f"{room.layout.floor_area_m2.value:.1f} sq m "
                  f"(connected={room.connected})")
        for w in property_layout.warnings:
            print(f"[roomscan] warning: {w}")
    except Exception as exc:
        from roomscan.render.svg_plan import render_stub
        render_stub(out / "plan.svg", f"Photo tier error: {exc}")
        print(f"[roomscan] photo tier failed: {exc}")


def _print_layout_summary(layout) -> None:
    print(f"[roomscan] floor area: {layout.floor_area_m2.value:.1f} sq m")
    if layout.ceiling_height_m:
        tag = " (unobserved - wide estimate)" if layout.ceiling_unobserved else ""
        print(f"[roomscan] ceiling height: {layout.ceiling_height_m.value:.2f} m{tag}")
    for w in layout.capture_warnings:
        print(f"[roomscan] warning: {w}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="roomscan", description="Floor plan reconstruction pipeline")
    p.add_argument("input_path", type=Path, help="Scan directory, video file, or photo folder")
    p.add_argument("--out", type=Path, default=Path("out"), help="Output directory (default: out/)")
    p.add_argument("--tier", choices=["lidar", "video", "photo"], default=None,
                   help="Force capture tier (default: auto-detect)")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    tier = args.tier or _detect_tier(args.input_path)
    print(f"[roomscan] tier: {tier}")

    if tier == "lidar":
        _run_lidar(args.input_path, args.out)
    elif tier == "video":
        _run_video(args.input_path, args.out)
    else:
        _run_photo(args.input_path, args.out)

    print(f"[roomscan] -> {args.out}/plan.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
