"""Video frame sampler for the video tier (Phase 7).

A generic handheld .mp4/.mov has no depth or pose metadata -- only RGB
frames. Sampling is time-based (not every-Nth-frame) so the cadence is
independent of the source frame rate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from roomscan.config import VIDEO_FRAME_INTERVAL_S, VIDEO_MAX_FRAMES


def sample_frames(
    video_path: Path,
    interval_s: float = VIDEO_FRAME_INTERVAL_S,
    max_frames: int = VIDEO_MAX_FRAMES,
) -> list[np.ndarray]:
    """Sample BGR frames from a video at a fixed time interval.

    Args:
        video_path: Path to a .mp4/.mov file.
        interval_s: Seconds between sampled frames.
        max_frames:  Cap on the number of frames returned (keeps SfM tractable).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            raise ValueError(f"Could not read video metadata from {video_path}")

        step = max(1, int(round(interval_s * fps)))
        frames: list[np.ndarray] = []
        idx = 0
        while idx < total and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            idx += step
        return frames
    finally:
        cap.release()
