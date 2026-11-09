"""Face swapping with the InsightFace inswapper_128 ONNX model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from face_detector import PROJECT_DIR, select_providers
from image_processor import paste_back

INSWAPPER_PATH = PROJECT_DIR / "models" / "inswapper_128.onnx"


class FaceSwapper:
    def __init__(self, model_path: Path = INSWAPPER_PATH) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Face-swap model not found: {model_path}\nRun:  python download_models.py"
            )
        import insightface

        self.model = insightface.model_zoo.get_model(str(model_path), providers=select_providers())

    def swap(self, target_img: np.ndarray, target_face, source_face) -> np.ndarray:
        """Replace `target_face` in `target_img` with the identity of `source_face`.

        Returns a new image at the original target resolution; the input is not modified.
        """
        if source_face is None or target_face is None:
            raise ValueError("Both a source face and a target face are required")
        fake128, M = self.model.get(target_img, target_face, source_face, paste_back=False)
        return paste_back(target_img, fake128, M, target_face)
