"""Live Webcam Face Swap - replace the face in av.jpg with your face from the webcam.

Run:  python app.py [--camera N] [--list-cameras] [--target av.jpg]

Controls (click the buttons or use the keys):
  C  Capture Face     R  Retake          S  Swap Face
  W  Save Result      N  Next camera     1-9  Select target face
  Q / Esc  Quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide probe noise while scanning cameras
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np

from image_processor import fit_into, load_target, save_result

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET = PROJECT_DIR / "av.jpg"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "av_face_swapped.jpg"
MODELS_DIR = PROJECT_DIR / "models"

WINDOW = "Face Swap - av.jpg"

# Layout
PANEL_W, PANEL_H = 460, 580
GAP = 10
HEADER_H = 34
BAR_H = 56
STATUS_H = 32
CANVAS_W = PANEL_W * 3 + GAP * 4
CANVAS_H = HEADER_H + PANEL_H + BAR_H + STATUS_H + GAP

BG = (24, 24, 24)
FG = (235, 235, 235)
GREEN = (80, 200, 80)
YELLOW = (60, 200, 230)
RED = (70, 70, 230)
GREY = (110, 110, 110)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def check_models() -> list[str]:
    missing = []
    if not (MODELS_DIR / "buffalo_l").exists() or not any((MODELS_DIR / "buffalo_l").glob("*.onnx")):
        missing.append("models/buffalo_l/")
    if not (MODELS_DIR / "inswapper_128.onnx").exists():
        missing.append("models/inswapper_128.onnx")
    return missing


class Button:
    def __init__(self, label: str, key: str, action, enabled_fn) -> None:
        self.label, self.key, self.action, self.enabled_fn = label, key, action, enabled_fn
        self.rect = (0, 0, 0, 0)

    def hit(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


class FaceSwapApp:
    def __init__(self, target_path: Path, camera_index: int | None) -> None:
        from face_detector import FaceDetector
        from face_swapper import FaceSwapper
        from webcam import Webcam, list_cameras

        self.target_path = target_path
        self.status = "Starting..."
        self.status_color = FG

        print("Loading face analysis models...")
        self.detector = FaceDetector()
        print(f"  providers: {self.detector.providers}")
        print("Loading face swap model...")
        self.swapper = FaceSwapper()

        # ---- Step 1: target image
        self.target = load_target(target_path)
        self.target_faces = self.detector.detect(self.target)
        self.selected_idx: int | None = None
        print(f"Target {target_path.name}: {self.target.shape[1]}x{self.target.shape[0]}, "
              f"{len(self.target_faces)} face(s) detected")

        # ---- Step 2: webcam
        print("Detecting webcams...")
        self.cameras = list_cameras()
        if not self.cameras:
            from webcam import PERMISSION_HINT
            raise RuntimeError("No working webcam found.\n" + PERMISSION_HINT)
        for c in self.cameras:
            print(f"  camera #{c['index']}: {c['width']}x{c['height']}")
        indices = [c["index"] for c in self.cameras]
        if camera_index is not None and camera_index not in indices:
            raise RuntimeError(f"Camera #{camera_index} not available. Found: {indices}")
        self._Webcam = Webcam
        self.cam_idx = camera_index if camera_index is not None else indices[0]
        self.cam = Webcam(self.cam_idx)
        print(f"Using camera #{self.cam_idx} ({self.cam.size[0]}x{self.cam.size[1]})")

        # ---- state
        self.live_frame: np.ndarray | None = None
        self.live_boxes = np.zeros((0, 5))
        self.frame_no = 0
        self.captured: np.ndarray | None = None
        self.source_face = None
        self.result: np.ndarray | None = None
        self.saved = False
        self.running = True
        self._quit_armed_until = 0.0
        self.target_panel_geom = (0, 0, 1.0, (0, 0))  # panel_x, panel_y, scale, offset

        self.buttons = [
            Button("Capture Face", "C", self.capture, lambda: self.captured is None and self.live_frame is not None),
            Button("Retake", "R", self.retake, lambda: self.captured is not None),
            Button("Swap Face", "S", self.swap,
                   lambda: self.source_face is not None and self.selected_idx is not None),
            Button("Save Result", "W", self.save, lambda: self.result is not None),
            Button("Quit", "Q", self.quit, lambda: True),
        ]

        if not self.target_faces:
            self.set_status(f"No face detected in {target_path.name} - swapping is not possible", RED)
        elif len(self.target_faces) == 1:
            self.selected_idx = 0
            self.set_status("Target face selected. Look at the camera and press Capture Face (C)", FG)
        else:
            self.set_status(f"{len(self.target_faces)} faces in {target_path.name}: click one "
                            f"(or press 1-{min(9, len(self.target_faces))}) to choose the target", YELLOW)

    # ------------------------------------------------------------- actions
    def set_status(self, text: str, color=FG) -> None:
        self.status, self.status_color = text, color
        print(f"[status] {text}")

    def capture(self) -> None:
        if self.live_frame is None:
            return
        frame = self.live_frame.copy()
        faces = self.detector.detect(frame)
        q = self.detector.validate_source_face(frame, faces)
        if not q.ok:
            self.set_status(f"Capture rejected: {q.reason}", RED)
            return
        self.captured = frame
        self.source_face = q.face
        self.result = None
        self.saved = False
        msg = "Face captured. " + q.reason + ". "
        msg += "Press Swap Face (S)" if self.selected_idx is not None else "Now select the target face"
        self.set_status(msg, GREEN)

    def retake(self) -> None:
        self.captured = None
        self.source_face = None
        self.result = None
        self.saved = False
        self.set_status("Live preview - press Capture Face (C) when ready", FG)

    def swap(self) -> None:
        if self.source_face is None:
            self.set_status("Capture your face first", RED)
            return
        if self.selected_idx is None:
            self.set_status("Select the target face in av.jpg first", RED)
            return
        self.set_status("Swapping face...", YELLOW)
        self.render()
        cv2.waitKey(1)
        t0 = time.time()
        try:
            self.result = self.swapper.swap(self.target, self.target_faces[self.selected_idx], self.source_face)
        except Exception as exc:  # show the error in the UI rather than crashing
            self.set_status(f"Swap failed: {exc}", RED)
            return
        assert self.result.shape == self.target.shape
        self.saved = False
        self.set_status(f"Swap done in {time.time() - t0:.1f}s "
                        f"({self.result.shape[1]}x{self.result.shape[0]}). Press Save Result (W)", GREEN)

    def save(self) -> None:
        if self.result is None:
            return
        try:
            path = save_result(self.result, OUTPUT_PATH, protected=[self.target_path])
        except Exception as exc:
            self.set_status(f"Save failed: {exc}", RED)
            return
        self.saved = True
        self.set_status(f"Saved: {path.relative_to(PROJECT_DIR)}", GREEN)

    def quit(self) -> None:
        self.running = False

    def request_quit(self) -> None:
        """Quit button needs a second click within 3 s, so a stray click can't close the app."""
        now = time.time()
        if now < self._quit_armed_until:
            self.quit()
        else:
            self._quit_armed_until = now + 3.0
            self.set_status("Click Quit again (or press Q) to exit", YELLOW)

    def select_target(self, idx: int) -> None:
        if 0 <= idx < len(self.target_faces):
            self.selected_idx = idx
            self.result = None
            self.set_status(f"Target face #{idx + 1} selected", GREEN)

    def next_camera(self) -> None:
        if len(self.cameras) < 2:
            self.set_status("Only one camera available", YELLOW)
            return
        indices = [c["index"] for c in self.cameras]
        nxt = indices[(indices.index(self.cam_idx) + 1) % len(indices)]
        self.cam.release()
        self.cam_idx = nxt
        self.cam = self._Webcam(nxt)
        self.retake()
        self.set_status(f"Switched to camera #{nxt}", FG)

    # ------------------------------------------------------------- input
    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for b in self.buttons:
            if b.hit(x, y):
                print(f"[click] ({x},{y}) -> {b.label} ({'enabled' if b.enabled_fn() else 'disabled'})")
                if b.enabled_fn():
                    if b.action == self.quit:
                        self.request_quit()
                    else:
                        b.action()
                return
        print(f"[click] ({x},{y}) -> no button")
        # Click on a target face box?
        px, py, scale, (ox, oy) = self.target_panel_geom
        ix, iy = (x - px - ox) / scale, (y - py - oy) / scale
        for i, f in enumerate(self.target_faces):
            x1, y1, x2, y2 = f.bbox
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                self.select_target(i)
                return

    def on_key(self, key: int) -> None:
        if key in (ord("q"), ord("Q"), 27):
            print("[exit] quit key pressed")
            self.quit()
            return
        if key in (ord("n"), ord("N")):
            self.next_camera()
            return
        if ord("1") <= key <= ord("9"):
            self.select_target(key - ord("1"))
            return
        for b in self.buttons:
            if key in (ord(b.key.lower()), ord(b.key)) and b.enabled_fn():
                b.action()
                return

    # ------------------------------------------------------------- drawing
    @staticmethod
    def _text(img, text, org, scale=0.55, color=FG, thick=1) -> None:
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)

    def _panel(self, canvas, idx: int, title: str, img: np.ndarray | None, placeholder: str):
        px = GAP + idx * (PANEL_W + GAP)
        py = HEADER_H
        self._text(canvas, title, (px + 4, HEADER_H - 10), 0.6)
        if img is None:
            panel = np.full((PANEL_H, PANEL_W, 3), (40, 40, 40), np.uint8)
            (tw, _), _ = cv2.getTextSize(placeholder, FONT, 0.55, 1)
            self._text(panel, placeholder, ((PANEL_W - tw) // 2, PANEL_H // 2), 0.55, GREY)
            scale, off = 1.0, (0, 0)
        else:
            panel, scale, off = fit_into(img, PANEL_W, PANEL_H, bg=(40, 40, 40))
        canvas[py:py + PANEL_H, px:px + PANEL_W] = panel
        return px, py, scale, off

    def render(self) -> None:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)

        # Panel 1: original av.jpg with face boxes
        orig = self.target.copy()
        lw = max(2, orig.shape[1] // 300)
        for i, f in enumerate(self.target_faces):
            x1, y1, x2, y2 = f.bbox.astype(int)
            color = GREEN if i == self.selected_idx else YELLOW
            cv2.rectangle(orig, (x1, y1), (x2, y2), color, lw)
            if len(self.target_faces) > 1:
                cv2.putText(orig, str(i + 1), (x1, max(30, y1 - 10)), FONT, lw * 0.6, color, lw, cv2.LINE_AA)
        self.target_panel_geom = self._panel(canvas, 0, f"Original: {self.target_path.name}", orig, "")

        # Panel 2: live preview (mirrored) or captured face
        if self.captured is not None:
            cap = self.captured.copy()
            x1, y1, x2, y2 = self.source_face.bbox.astype(int)
            cv2.rectangle(cap, (x1, y1), (x2, y2), GREEN, 3)
            self._panel(canvas, 1, "Captured webcam face", cv2.flip(cap, 1), "")
        elif self.live_frame is not None:
            live = self.live_frame.copy()
            for b in self.live_boxes:
                x1, y1, x2, y2 = b[:4].astype(int)
                cv2.rectangle(live, (x1, y1), (x2, y2), GREEN if len(self.live_boxes) == 1 else YELLOW, 2)
            live = cv2.flip(live, 1)
            hint = ("Face detected" if len(self.live_boxes) == 1
                    else "No face" if len(self.live_boxes) == 0 else f"{len(self.live_boxes)} faces")
            self._text(live, f"LIVE cam #{self.cam_idx} - {hint}", (14, 36), 1.0,
                       GREEN if len(self.live_boxes) == 1 else RED, 2)
            self._panel(canvas, 1, "Live webcam", live, "")
        else:
            self._panel(canvas, 1, "Live webcam", None, "Waiting for camera...")

        # Panel 3: result
        title = "Face-swapped result" + (" (saved)" if self.saved else "")
        self._panel(canvas, 2, title, self.result, "Capture + Swap to see the result")

        # Buttons
        by = HEADER_H + PANEL_H + GAP
        bw = (CANVAS_W - GAP * (len(self.buttons) + 1)) // len(self.buttons)
        for i, b in enumerate(self.buttons):
            bx = GAP + i * (bw + GAP)
            b.rect = (bx, by, bx + bw, by + BAR_H - 12)
            enabled = b.enabled_fn()
            fill = (70, 110, 60) if enabled else (55, 55, 55)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + BAR_H - 12), fill, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + BAR_H - 12), (150, 150, 150) if enabled else GREY, 1)
            label = f"{b.label} [{b.key}]"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
            self._text(canvas, label, (bx + (bw - tw) // 2, by + (BAR_H - 12 + th) // 2), 0.6,
                       FG if enabled else GREY)

        # Status line
        sy = HEADER_H + PANEL_H + BAR_H + GAP + 18
        self._text(canvas, self.status, (GAP, sy), 0.55, self.status_color)

        cv2.imshow(WINDOW, canvas)

    # ------------------------------------------------------------- loop
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.render()
        cv2.waitKey(1)
        print(f"[window] canvas {CANVAS_W}x{CANVAS_H}, image rect {cv2.getWindowImageRect(WINDOW)}")
        for b in self.buttons:
            print(f"[window] button {b.label!r} at {b.rect}")
        try:
            while self.running:
                if self.captured is None:
                    frame = self.cam.read()
                    if frame is not None:
                        self.live_frame = frame
                        if self.frame_no % 3 == 0:
                            try:
                                self.live_boxes = self.detector.detect_boxes(frame)
                            except Exception:
                                self.live_boxes = np.zeros((0, 5))
                        self.frame_no += 1
                self.render()
                key = cv2.waitKey(15) & 0xFF
                if key != 255:
                    self.on_key(key)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    print("[exit] window closed")
                    break
        finally:
            self.cam.release()
            cv2.destroyAllWindows()
            for _ in range(5):  # let macOS actually close the window
                cv2.waitKey(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live webcam face swap into av.jpg")
    parser.add_argument("--camera", type=int, default=None, help="webcam device index (default: first working)")
    parser.add_argument("--list-cameras", action="store_true", help="list detected webcams and exit")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="target image (default: av.jpg)")
    args = parser.parse_args()

    if args.list_cameras:
        from webcam import list_cameras
        cams = list_cameras()
        if not cams:
            print("No working webcam found.")
            return 1
        for c in cams:
            print(f"camera #{c['index']}: {c['width']}x{c['height']}")
        return 0

    target = args.target if args.target.is_absolute() else (Path.cwd() / args.target)
    if not target.exists():
        target = PROJECT_DIR / args.target.name
    if not target.exists():
        print(f"Error: target image '{args.target}' not found. Place av.jpg in {PROJECT_DIR}")
        return 1

    missing = check_models()
    if missing:
        print("Error: missing model files: " + ", ".join(missing))
        print("Run:  python download_models.py   (read and accept the model licence first)")
        return 1

    try:
        app = FaceSwapApp(target.resolve(), args.camera)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
