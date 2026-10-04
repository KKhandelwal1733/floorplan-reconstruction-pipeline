"""CLI entry point — each phase wires in its stage."""
import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout: a default Windows console (cp1252) otherwise raises
# UnicodeEncodeError on the first non-ASCII character and crashes the run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
DAMAGE_SCAN_N_FRAMES = 15   # frames sampled for damage detection (lidar/video tiers)
DAMAGE_SCAN_INTERVAL_S = 2.0


def _detect_tier(input_path: Path) -> str:
    """Auto-detect capture tier from the input path shape."""
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    if input_path.is_dir() and (input_path / "depth").exists():
        return "lidar"
    if input_path.is_dir():
        return "photo"
    raise ValueError(f"Could not detect capture tier for {input_path}")


def _run_damage_scan(frames, layout):
    """Shared by all tiers: never let damage detection itself crash a run
    that otherwise succeeded at geometry -- report 0 findings and a warning
    instead."""
    from roomscan.damage.pipeline import detect_damage_for_room
    try:
        damages, scope_items = detect_damage_for_room(frames, layout)
    except Exception as exc:
        print(f"[roomscan] damage detection failed (geometry unaffected): {exc}")
        return [], []
    print(f"[roomscan] {len(damages)} damage candidate(s) flagged (advisory, crude heuristic; "
          "human review required -- see COMPLIANCE.md)")
    for d in damages[:5]:
        print(f"[roomscan]   {d.class_} on {d.surface_id}: ~{d.area_m2.value:.2f} m2")
    if len(damages) > 5:
        print(f"[roomscan]   ... and {len(damages) - 5} more")
    return damages, scope_items


def _run_capture_quality_check(frames, layout) -> None:
    """Shared by all tiers: flag low light / mirror / glass / wet-look
    surfaces in submitted frames (advisory, crude heuristic -- never lets a
    failure here affect an otherwise-successful geometry run)."""
    from roomscan.geometry.capture_quality import apply_capture_quality_warnings
    try:
        issues = apply_capture_quality_warnings(layout, frames)
    except Exception as exc:
        print(f"[roomscan] capture-quality check failed (geometry unaffected): {exc}")
        return
    for w in issues:
        print(f"[roomscan] warning: {w}")


def _is_scan_dir(d: Path) -> bool:
    return (d / "depth").exists()


def _find_video_file(room_dir: Path) -> Path | None:
    for p in sorted(room_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            return p
    return None


def _run_lidar(input_path: Path, out: Path) -> None:
    """A single Stray Scanner export folder (has depth/ directly), or a
    property folder of several such exports (one per room) -- the latter is
    stitched via the door-to-door pose graph, same as the photo tier."""
    if _is_scan_dir(input_path):
        _run_lidar_single(input_path, out)
    else:
        _run_lidar_property(input_path, out)


def _run_lidar_single(input_path: Path, out: Path) -> None:
    print(f"[roomscan] loading scan: {input_path}")
    from roomscan.io.stray_scanner import load_scan
    pts = load_scan(input_path, max_frames=None)
    print(f"[roomscan] {len(pts):,} points loaded")

    print("[roomscan] extracting layout...")
    try:
        from roomscan.geometry.openings import detect_openings
        from roomscan.geometry.room_layout import extract_layout
        from roomscan.plan_builder import write_single_room_plan
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

        damages, scope_items = [], []
        video_path = input_path / "rgb.mp4"
        if video_path.exists():
            from roomscan.io.video_loader import sample_frames
            frames = sample_frames(video_path, interval_s=DAMAGE_SCAN_INTERVAL_S,
                                    max_frames=DAMAGE_SCAN_N_FRAMES)
            _run_capture_quality_check(frames, layout)
            damages, scope_items = _run_damage_scan(frames, layout)

        write_single_room_plan("lidar", layout, out / "plan.json", damages, scope_items, openings)
    except Exception as exc:
        from roomscan.plan_builder import write_abstained_plan
        from roomscan.render.svg_plan import render_stub
        render_stub(out / "plan.svg", f"Layout error: {exc}")
        write_abstained_plan("lidar", f"Layout error: {exc}", out / "plan.json")
        print(f"[roomscan] layout failed: {exc}")


def _run_lidar_property(input_path: Path, out: Path) -> None:
    print(f"[roomscan] loading property folder (lidar): {input_path}")
    try:
        from roomscan.geometry.multi_room import stitch_rooms
        from roomscan.geometry.openings import detect_openings
        from roomscan.geometry.room_layout import extract_layout
        from roomscan.io.photo_loader import list_room_dirs
        from roomscan.io.stray_scanner import load_scan
        from roomscan.io.video_loader import sample_frames
        from roomscan.plan_builder import write_property_plan
        from roomscan.render.svg_plan import render_property

        room_dirs = [d for d in list_room_dirs(input_path) if _is_scan_dir(d)]
        results: list[tuple[str, object, list]] = []
        openings_per_room: list[list] = []
        per_room_damage = {}
        failures: dict[str, str] = {}

        for room_dir in room_dirs:
            try:
                pts = load_scan(room_dir, max_frames=None)
                layout = extract_layout(pts)
                openings = detect_openings(pts, layout)
                results.append((room_dir.name, layout, []))
                openings_per_room.append(openings)

                video_path = room_dir / "rgb.mp4"
                damages, scope_items = [], []
                if video_path.exists():
                    frames = sample_frames(video_path, interval_s=DAMAGE_SCAN_INTERVAL_S,
                                            max_frames=DAMAGE_SCAN_N_FRAMES)
                    _run_capture_quality_check(frames, layout)
                    damages, scope_items = _run_damage_scan(frames, layout)
                per_room_damage[room_dir.name] = (damages, scope_items)
            except Exception as e:
                failures[room_dir.name] = str(e)

        property_layout = stitch_rooms(results, openings_per_room)
        for name, reason in failures.items():
            property_layout.warnings.append(f"room '{name}' skipped: {reason}")
        render_property(property_layout, out / "plan.svg")
        for w in property_layout.warnings:
            print(f"[roomscan] warning: {w}")
        print(f"[roomscan] {len(results)}/{len(room_dirs)} room(s) reconstructed")

        write_property_plan(property_layout, out / "plan.json", per_room_damage, tier="lidar")
    except Exception as exc:
        from roomscan.plan_builder import write_abstained_plan
        from roomscan.render.svg_plan import render_stub
        render_stub(out / "plan.svg", f"Lidar property error: {exc}")
        write_abstained_plan("lidar", f"Lidar property error: {exc}", out / "plan.json")
        print(f"[roomscan] lidar property tier failed: {exc}")


def _run_video(input_path: Path, out: Path) -> None:
    """A single handheld video file, or a property folder of per-room
    subfolders (one video per room) -- stitched via the door-to-door pose
    graph, same as the lidar/photo tiers."""
    if input_path.is_dir():
        _run_video_property(input_path, out)
    else:
        _run_video_single(input_path, out)


def _run_video_single(input_path: Path, out: Path) -> None:
    print(f"[roomscan] loading video: {input_path}")
    try:
        from roomscan.geometry.video_tier import process_video
        from roomscan.io.video_loader import sample_frames
        from roomscan.plan_builder import write_single_room_plan
        from roomscan.render.svg_plan import render_plan
        layout, diagnostics = process_video(input_path)
        render_plan(layout, out / "plan.svg")
        _print_layout_summary(layout)
        print(f"[roomscan] video tier: {diagnostics['n_frames_sampled']} frames sampled, "
              f"{diagnostics['n_points_unscaled']} sparse points reconstructed, "
              f"scale +/-{diagnostics['scale_rel_half_width']:.0%}")

        frames = sample_frames(input_path, interval_s=DAMAGE_SCAN_INTERVAL_S,
                                max_frames=DAMAGE_SCAN_N_FRAMES)
        _run_capture_quality_check(frames, layout)
        damages, scope_items = _run_damage_scan(frames, layout)

        write_single_room_plan("video", layout, out / "plan.json", damages, scope_items)
    except Exception as exc:
        from roomscan.plan_builder import write_abstained_plan
        from roomscan.render.svg_plan import render_stub
        render_stub(out / "plan.svg", f"Video reconstruction error: {exc}")
        write_abstained_plan("video", f"Video reconstruction error: {exc}", out / "plan.json")
        print(f"[roomscan] video tier failed: {exc}")


def _run_video_property(input_path: Path, out: Path) -> None:
    print(f"[roomscan] loading property folder (video): {input_path}")
    try:
        from roomscan.geometry.multi_room import stitch_rooms
        from roomscan.geometry.openings import detect_openings
        from roomscan.geometry.video_tier import process_video
        from roomscan.io.photo_loader import list_room_dirs
        from roomscan.io.video_loader import sample_frames
        from roomscan.plan_builder import write_property_plan
        from roomscan.render.svg_plan import render_property

        room_dirs = list_room_dirs(input_path)
        results: list[tuple[str, object, list]] = []
        openings_per_room: list[list] = []
        per_room_damage = {}
        failures: dict[str, str] = {}

        for room_dir in room_dirs:
            video_path = _find_video_file(room_dir)
            if video_path is None:
                failures[room_dir.name] = "no video file found in room folder"
                continue
            try:
                layout, diagnostics = process_video(video_path)
                openings = detect_openings(diagnostics["pts_scaled"], layout)
                results.append((room_dir.name, layout, []))
                openings_per_room.append(openings)

                frames = sample_frames(video_path, interval_s=DAMAGE_SCAN_INTERVAL_S,
                                        max_frames=DAMAGE_SCAN_N_FRAMES)
                _run_capture_quality_check(frames, layout)
                per_room_damage[room_dir.name] = _run_damage_scan(frames, layout)
            except Exception as e:
                failures[room_dir.name] = str(e)

        property_layout = stitch_rooms(results, openings_per_room)
        for name, reason in failures.items():
            property_layout.warnings.append(f"room '{name}' skipped: {reason}")
        render_property(property_layout, out / "plan.svg")
        for w in property_layout.warnings:
            print(f"[roomscan] warning: {w}")
        print(f"[roomscan] {len(results)}/{len(room_dirs)} room(s) reconstructed")

        write_property_plan(property_layout, out / "plan.json", per_room_damage, tier="video")
    except Exception as exc:
        from roomscan.plan_builder import write_abstained_plan
        from roomscan.render.svg_plan import render_stub
        render_stub(out / "plan.svg", f"Video property error: {exc}")
        write_abstained_plan("video", f"Video property error: {exc}", out / "plan.json")
        print(f"[roomscan] video property tier failed: {exc}")


def _run_photo(input_path: Path, out: Path) -> None:
    print(f"[roomscan] loading property folder: {input_path}")
    try:
        from roomscan.geometry.photo_tier import process_property
        from roomscan.io.photo_loader import load_photos
        from roomscan.plan_builder import write_property_plan
        from roomscan.render.svg_plan import render_property
        property_layout, diagnostics = process_property(input_path)
        render_property(property_layout, out / "plan.svg")
        print(f"[roomscan] {diagnostics['n_rooms_reconstructed']}/{diagnostics['n_rooms_found']} "
              f"room(s) reconstructed")

        per_room_damage = {}
        for room in property_layout.rooms:
            print(f"[roomscan]   room '{room.name}': "
                  f"{room.layout.floor_area_m2.value:.1f} sq m "
                  f"(connected={room.connected})")
            photos = load_photos(input_path / room.name)
            _run_capture_quality_check(photos, room.layout)
            per_room_damage[room.name] = _run_damage_scan(photos, room.layout)
        for w in property_layout.warnings:
            print(f"[roomscan] warning: {w}")

        write_property_plan(property_layout, out / "plan.json", per_room_damage)
    except Exception as exc:
        from roomscan.plan_builder import write_abstained_plan
        from roomscan.render.svg_plan import render_stub
        render_stub(out / "plan.svg", f"Photo tier error: {exc}")
        write_abstained_plan("photo", f"Photo tier error: {exc}", out / "plan.json")
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
    print(f"[roomscan] -> {args.out}/plan.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
