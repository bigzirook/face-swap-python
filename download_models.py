"""Download the model weights required by the face-swap app (after licence confirmation).

Models
------
1. buffalo_l  (InsightFace detection / landmarks / recognition pack)
   Source : https://github.com/deepinsight/insightface/releases (auto-download by insightface)
   Licence: InsightFace pretrained models are for NON-COMMERCIAL RESEARCH PURPOSES ONLY.

2. inswapper_128.onnx  (InsightFace face-swap model)
   InsightFace no longer hosts this file officially; it is fetched from a widely used
   Hugging Face mirror and verified against its known SHA-256.
   Licence: NON-COMMERCIAL RESEARCH PURPOSES ONLY (same InsightFace model licence).

Usage:  python download_models.py [--yes]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

INSWAPPER_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
INSWAPPER_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
INSWAPPER_PATH = MODELS_DIR / "inswapper_128.onnx"

LICENCE_NOTICE = """
======================================================================
 MODEL LICENCE NOTICE
----------------------------------------------------------------------
 This app uses pretrained models from the InsightFace project:
   * buffalo_l           (face detection, landmarks, recognition)
   * inswapper_128.onnx  (face swapping)

 InsightFace states that its pretrained models are available for
 NON-COMMERCIAL RESEARCH PURPOSES ONLY. Commercial use requires a
 separate licence from InsightFace (https://insightface.ai).

 You are responsible for only using images of people who have given
 their permission, and for not using results to deceive or impersonate.
======================================================================
"""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_inswapper() -> None:
    if INSWAPPER_PATH.exists():
        print(f"[ok] {INSWAPPER_PATH.name} already present - verifying checksum...")
        if sha256_of(INSWAPPER_PATH) == INSWAPPER_SHA256:
            print("[ok] checksum matches")
            return
        print("[!] checksum mismatch - re-downloading")
        INSWAPPER_PATH.unlink()

    tmp = INSWAPPER_PATH.with_suffix(".onnx.part")
    print(f"Downloading inswapper_128.onnx (~554 MB) from:\n  {INSWAPPER_URL}")
    with requests.get(INSWAPPER_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    digest = sha256_of(tmp)
    if digest != INSWAPPER_SHA256:
        tmp.unlink(missing_ok=True)
        sys.exit(f"[x] SHA-256 mismatch ({digest}). File discarded.")
    tmp.rename(INSWAPPER_PATH)
    print("[ok] inswapper_128.onnx downloaded and verified")


def download_buffalo() -> None:
    target = MODELS_DIR / "buffalo_l"
    if target.exists() and any(target.glob("*.onnx")):
        print("[ok] buffalo_l already present")
        return
    print("Downloading buffalo_l via insightface (~280 MB)...")
    from insightface.utils.storage import ensure_available

    ensure_available("models", "buffalo_l", root=str(PROJECT_DIR))
    print("[ok] buffalo_l downloaded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="accept the licence notice non-interactively")
    args = parser.parse_args()

    print(LICENCE_NOTICE)
    if not args.yes:
        answer = input("Type 'yes' to accept these terms and download the models: ").strip().lower()
        if answer != "yes":
            sys.exit("Aborted - no models downloaded.")

    MODELS_DIR.mkdir(exist_ok=True)
    download_buffalo()
    download_inswapper()
    print("\nAll models ready. Run:  python app.py")


if __name__ == "__main__":
    main()
