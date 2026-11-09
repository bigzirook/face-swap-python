"""Image loading, face blending (colour matching + soft masks) and saving."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

PROTECTED_NAMES = {"av.jpg"}


# ---------------------------------------------------------------------- I/O
def load_target(path: str | Path) -> np.ndarray:
    """Load the target image read-only (EXIF orientation applied) as BGR uint8."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target image not found: {path}")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.array(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_result(img_bgr: np.ndarray, out_path: str | Path, protected: list[Path] | None = None) -> Path:
    """Save as high-quality JPEG. Refuses to overwrite the original target image."""
    out_path = Path(out_path).resolve()
    for p in protected or []:
        if out_path == Path(p).resolve():
            raise PermissionError(f"Refusing to overwrite original image {p}")
    if out_path.name in PROTECTED_NAMES and out_path.parent.name != "outputs":
        raise PermissionError(f"Refusing to overwrite original image {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(out_path, "JPEG", quality=95, subsampling=0)
    return out_path


# ---------------------------------------------------------------- blending
def _odd(n: float) -> int:
    n = max(int(n), 1)
    return n if n % 2 == 1 else n + 1


def _build_mask(shape: tuple[int, int], M: np.ndarray, face, face_size: float) -> np.ndarray:
    """Soft float mask (0..1) of the region that receives the swapped face."""
    h, w = shape
    IM = cv2.invertAffineTransform(M)

    # 1) The swapped 128x128 crop region, warped back to the target
    box = np.full((128, 128), 255, np.uint8)
    box_mask = cv2.warpAffine(box, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    erode_k = _odd(face_size * 0.06)
    box_mask = cv2.erode(box_mask, np.ones((erode_k, erode_k), np.uint8), iterations=1)

    # 2) Face-shape mask from the 106 landmarks, extended upward to cover the forehead
    lmk = getattr(face, "landmark_2d_106", None)
    if lmk is not None:
        pts = lmk.astype(np.float32)
        center = pts.mean(axis=0)
        top = pts[:, 1].min()
        # Push points above the eye line further up (forehead) by ~18% of face height
        eye_line = center[1] - 0.05 * face_size
        lifted = pts.copy()
        upper = lifted[:, 1] < eye_line
        lifted[upper, 1] -= 0.18 * face_size * (eye_line - lifted[upper, 1]) / max(eye_line - top, 1)
        hull = cv2.convexHull(np.concatenate([pts, lifted]).astype(np.int32))
        shape_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(shape_mask, hull, 255)
        dil = _odd(face_size * 0.04)
        shape_mask = cv2.dilate(shape_mask, np.ones((dil, dil), np.uint8))
        mask = cv2.bitwise_and(box_mask, shape_mask)
    else:
        mask = box_mask

    # 3) Feather the boundary
    shrink = _odd(face_size * 0.03)
    mask = cv2.erode(mask, np.ones((shrink, shrink), np.uint8))
    blur = _odd(face_size * 0.10)
    soft = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    return soft


def match_color(src: np.ndarray, ref: np.ndarray, mask: np.ndarray, strength: float = 0.9) -> np.ndarray:
    """Reinhard colour transfer in LAB, statistics computed only inside `mask`."""
    region = mask > 0.5
    if region.sum() < 100:
        return src
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src_lab.copy()
    for c in range(3):
        s = src_lab[..., c][region]
        r = ref_lab[..., c][region]
        s_mu, s_sd = s.mean(), s.std() + 1e-6
        r_mu, r_sd = r.mean(), r.std() + 1e-6
        # Limit std scaling so contrast is not blown out
        scale = float(np.clip(r_sd / s_sd, 0.6, 1.6))
        transferred = (src_lab[..., c] - s_mu) * scale + r_mu
        out[..., c] = src_lab[..., c] * (1 - strength) + transferred * strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _unsharp(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def paste_back(target: np.ndarray, fake128: np.ndarray, M: np.ndarray, face) -> np.ndarray:
    """Warp the 128px swapped face back into `target` with colour matching and soft blending.

    Works at the full target resolution; `target` itself is not modified.
    """
    h, w = target.shape[:2]
    x1, y1, x2, y2 = face.bbox
    face_size = float(max(x2 - x1, y2 - y1))

    IM = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(fake128, IM, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    mask = _build_mask((h, w), M, face, face_size)

    # Work inside the face's bounding region only (faster, avoids touching the rest)
    ys, xs = np.where(mask > 0.001)
    if len(xs) == 0:
        return target.copy()
    pad = int(face_size * 0.05)
    rx1, rx2 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    ry1, ry2 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)

    t_roi = target[ry1:ry2, rx1:rx2]
    f_roi = warped[ry1:ry2, rx1:rx2]
    m_roi = mask[ry1:ry2, rx1:rx2]

    # Skin-tone / lighting match to the original face
    f_roi = match_color(f_roi, t_roi, m_roi)

    # Counter the softness of upscaling a 128px face
    upscale = face_size / 128.0
    if upscale > 1.5:
        f_roi = _unsharp(f_roi, sigma=min(upscale * 0.6, 3.0), amount=0.25)

    alpha = m_roi[..., None]
    blended = f_roi.astype(np.float32) * alpha + t_roi.astype(np.float32) * (1 - alpha)

    out = target.copy()
    out[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------ display
def fit_into(img: np.ndarray, box_w: int, box_h: int, bg=(32, 32, 32)):
    """Letterbox `img` into a box. Returns (canvas, scale, (off_x, off_y))."""
    h, w = img.shape[:2]
    scale = min(box_w / w, box_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((box_h, box_w, 3), bg, np.uint8)
    ox, oy = (box_w - nw) // 2, (box_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, scale, (ox, oy)
