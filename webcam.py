"""Physical webcam access via OpenCV (macOS AVFoundation / Linux V4L2).

Only integer device indices are accepted - video files, URLs and other
non-device sources are deliberately rejected.
"""

from __future__ import annotations

import platform
import time

import cv2
import numpy as np


def _backend() -> int:
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


PERMISSION_HINT = (
    "Could not read from the camera.\n"
    "  macOS: System Settings > Privacy & Security > Camera > enable your terminal app "
    "(Terminal / iTerm / VS Code), then restart it.\n"
    "  Linux: make sure your user is in the 'video' group and no other app is using the camera."
)


def _try_open(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        cap.release()
        return None
    # Some cameras deliver a few empty frames while warming up
    for _ in range(20):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


def list_cameras(max_index: int = 6) -> list[dict]:
    """Probe device indices and return the cameras that deliver real frames."""
    found = []
    for idx in range(max_index):
        cap = _try_open(idx)
        if cap is None:
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        found.append({"index": idx, "width": w, "height": h})
        cap.release()
    return found


class Webcam:
    """Live physical webcam. Use as a context manager or call release()."""

    def __init__(self, index: int, width: int = 1280, height: int = 720) -> None:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("Webcam only accepts a physical device index (0, 1, ...)")
        self.index = index
        self.cap = cv2.VideoCapture(index, _backend())
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera #{index}.\n{PERMISSION_HINT}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        for _ in range(30):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if frame is None:
            self.release()
            raise RuntimeError(PERMISSION_HINT)
        self.size = (frame.shape[1], frame.shape[0])

    def read(self) -> np.ndarray | None:
        """Return the latest raw (non-mirrored) BGR frame, or None."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
