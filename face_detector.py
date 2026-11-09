"""Face detection, landmarks and quality validation using InsightFace (buffalo_l)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_DIR = Path(__file__).resolve().parent

# Quality thresholds for the captured webcam (source) face
MIN_DET_SCORE = 0.60
MIN_FACE_WIDTH = 100  # pixels
MAX_YAW = 30.0  # degrees
MAX_PITCH = 30.0
MIN_SHARPNESS = 40.0  # variance of Laplacian on the face crop
BRIGHTNESS_RANGE = (50, 210)


def select_providers() -> list[str]:
    """Choose ONNX Runtime execution providers.

    Default: CUDA if available, else CPU. CoreML can be enabled explicitly with
    FACESWAP_PROVIDERS=coreml (it is not used by default because some InsightFace
    models run unreliably on it).
    """
    available = ort.get_available_providers()
    wanted = os.environ.get("FACESWAP_PROVIDERS", "").lower()
    providers: list[str] = []
    if "coreml" in wanted and "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    if "CUDAExecutionProvider" in available and "cpu" not in wanted:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


@dataclass
class QualityResult:
    ok: bool
    reason: str
    face: object | None = None


class FaceDetector:
    def __init__(self, det_size: tuple[int, int] = (640, 640)) -> None:
        model_dir = PROJECT_DIR / "models" / "buffalo_l"
        if not model_dir.exists() or not any(model_dir.glob("*.onnx")):
            raise FileNotFoundError(
                f"Face analysis models not found in {model_dir}.\nRun:  python download_models.py"
            )
        from insightface.app import FaceAnalysis

        self.providers = select_providers()
        self.app = FaceAnalysis(name="buffalo_l", root=str(PROJECT_DIR), providers=self.providers)
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, img_bgr: np.ndarray) -> list:
        """Return all detected faces, sorted left-to-right."""
        faces = self.app.get(img_bgr)
        return sorted(faces, key=lambda f: f.bbox[0])

    def detect_boxes(self, img_bgr: np.ndarray) -> np.ndarray:
        """Fast detection only (no landmarks/recognition) for the live preview overlay."""
        bboxes, _kps = self.app.det_model.detect(img_bgr, max_num=0, metric="default")
        return bboxes  # (N, 5): x1, y1, x2, y2, score

    # ------------------------------------------------------------------ quality
    @staticmethod
    def _face_crop(img: np.ndarray, face) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return img[y1:y2, x1:x2]

    def validate_source_face(self, frame: np.ndarray, faces: list) -> QualityResult:
        """Check that a webcam frame contains exactly one good-quality face."""
        if not faces:
            return QualityResult(False, "No face detected - look at the camera")

        faces_by_area = sorted(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True
        )
        face = faces_by_area[0]
        if len(faces_by_area) > 1:
            a0 = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            b = faces_by_area[1].bbox
            a1 = (b[2] - b[0]) * (b[3] - b[1])
            if a1 > 0.5 * a0:
                return QualityResult(False, f"{len(faces)} faces in view - only you should be visible")

        if face.det_score < MIN_DET_SCORE:
            return QualityResult(False, f"Face detection confidence too low ({face.det_score:.2f})")

        width = face.bbox[2] - face.bbox[0]
        if width < MIN_FACE_WIDTH:
            return QualityResult(False, f"Face too small ({int(width)}px) - move closer")

        pose = getattr(face, "pose", None)
        if pose is not None:
            pitch, yaw, _roll = (float(v) for v in pose)
            if abs(yaw) > MAX_YAW:
                return QualityResult(False, f"Head turned too far ({yaw:.0f} deg) - face the camera")
            if abs(pitch) > MAX_PITCH:
                return QualityResult(False, f"Head tilted too far ({pitch:.0f} deg) - look straight")

        crop = self._face_crop(frame, face)
        if crop.size == 0:
            return QualityResult(False, "Face is partially outside the frame")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < MIN_SHARPNESS:
            return QualityResult(False, f"Image too blurry (sharpness {sharpness:.0f}) - hold still")
        brightness = float(gray.mean())
        if brightness < BRIGHTNESS_RANGE[0]:
            return QualityResult(False, "Face too dark - add more light")
        if brightness > BRIGHTNESS_RANGE[1]:
            return QualityResult(False, "Face overexposed - reduce direct light")

        return QualityResult(True, f"Good face (score {face.det_score:.2f}, sharpness {sharpness:.0f})", face)
