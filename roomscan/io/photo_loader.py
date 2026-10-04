"""Photo-folder loader for the photo tier (Phase 8).

A room's input is 2-8 unordered still images. There's no timestamp/pose
metadata to establish true capture order, so files are sorted by name --
a reasonable proxy for capture order with typical camera/phone naming
(IMG_0001.jpg, IMG_0002.jpg, ...), and good enough for reconstruct_sparse's
consecutive-pair matching to find overlapping views.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_photos(room_dir: Path) -> list[np.ndarray]:
    """Load all images in a room folder, sorted by filename, as BGR arrays."""
    import cv2

    room_dir = Path(room_dir)
    paths = sorted(
        p for p in room_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    photos = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            photos.append(img)
    return photos


def list_room_dirs(property_dir: Path) -> list[Path]:
    """Per-room subfolders of a multi-room property folder, sorted by name."""
    property_dir = Path(property_dir)
    return sorted(d for d in property_dir.iterdir() if d.is_dir())
