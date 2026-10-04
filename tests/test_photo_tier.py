"""Phase 8: photo tier tests.

Unlike the video tier (continuous frames, narrow baseline between samples),
2-8 unordered stills often have far too little shared visual overlap for
any pair to reconstruct at all -- confirmed empirically: simulated photo
sets derived from the real single_room video succeeded at n=12 (94% floor
area error vs LiDAR truth) but gracefully abstained (no crash, clear
ValueError) at n=8, 15, and 20 in testing. These tests therefore check
structural correctness (never crashes, degrades to abstention cleanly, and
succeeds-with-a-sane-type when it does succeed) rather than asserting that
reconstruction always succeeds, since that has not been demonstrated to be
reliable for this lightweight method.
"""
from pathlib import Path

import pytest

from roomscan.geometry.sfm import reconstruct_best_pair

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_ROOM_VIDEO = FIXTURES / "single_room" / "rgb.mp4"


def test_reconstruct_best_pair_empty_for_no_frames():
    pts, poses = reconstruct_best_pair([])
    assert len(pts) == 0
    assert poses == []


@pytest.mark.skipif(not SINGLE_ROOM_VIDEO.exists(), reason="single_room/rgb.mp4 fixture not present")
def test_process_room_photos_never_crashes(tmp_path):
    """Whatever the outcome (success or abstention), this must not raise an
    unhandled exception other than the documented ValueError."""
    import cv2
    from bench.derive_tiers import derive_simulated_photos
    from roomscan.geometry.photo_tier import process_room_photos

    room_dir = tmp_path / "room1"
    room_dir.mkdir()
    photos = derive_simulated_photos(SINGLE_ROOM_VIDEO, n_photos=6)
    for i, p in enumerate(photos):
        cv2.imwrite(str(room_dir / f"img_{i:02d}.jpg"), p)

    try:
        layout, diagnostics = process_room_photos(room_dir)
        assert layout.floor_area_m2.value > 0
        assert diagnostics["n_photos"] == len(photos)
        assert any("photo tier" in w for w in layout.capture_warnings)
    except ValueError:
        pass  # abstention is an acceptable, documented outcome


@pytest.mark.skipif(not SINGLE_ROOM_VIDEO.exists(), reason="single_room/rgb.mp4 fixture not present")
def test_process_room_photos_too_few_photos_raises(tmp_path):
    import cv2
    from bench.derive_tiers import derive_simulated_photos
    from roomscan.geometry.photo_tier import process_room_photos

    room_dir = tmp_path / "room1"
    room_dir.mkdir()
    photos = derive_simulated_photos(SINGLE_ROOM_VIDEO, n_photos=1)
    for i, p in enumerate(photos):
        cv2.imwrite(str(room_dir / f"img_{i:02d}.jpg"), p)

    with pytest.raises(ValueError, match="need >="):
        process_room_photos(room_dir)


@pytest.mark.skipif(not SINGLE_ROOM_VIDEO.exists(), reason="single_room/rgb.mp4 fixture not present")
def test_process_property_handles_mixed_success_and_failure(tmp_path):
    """A multi-room property where some rooms fail to reconstruct must still
    produce a valid (possibly partial) PropertyLayout, never crash."""
    import cv2
    from bench.derive_tiers import derive_simulated_photos
    from roomscan.geometry.photo_tier import process_property

    property_dir = tmp_path / "property"
    for name, n in [("living_room", 12), ("closet", 2)]:
        room_dir = property_dir / name
        room_dir.mkdir(parents=True)
        photos = derive_simulated_photos(SINGLE_ROOM_VIDEO, n_photos=n)
        for i, p in enumerate(photos):
            cv2.imwrite(str(room_dir / f"img_{i:02d}.jpg"), p)

    property_layout, diagnostics = process_property(property_dir)
    assert diagnostics["n_rooms_found"] == 2
    assert diagnostics["n_rooms_reconstructed"] + diagnostics["n_rooms_failed"] == 2
    assert len(property_layout.rooms) == diagnostics["n_rooms_reconstructed"]
    # every reconstructed room must carry a usable floor area
    for room in property_layout.rooms:
        assert room.layout.floor_area_m2.value > 0
