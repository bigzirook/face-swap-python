# Live Webcam Face Swap

Replaces the face in `av.jpg` with your face, captured live from your laptop's physical webcam. The target's body, clothes, pose, background and lighting stay unchanged. The result is saved to `outputs/av_face_swapped.jpg`. The original `av.jpg` is never modified.

Stack: Python 3.11, OpenCV (webcam + GUI), InsightFace (`buffalo_l` detection/landmarks/recognition + `inswapper_128` swap), ONNX Runtime, NumPy, Pillow.

## Setup

### macOS (Apple Silicon)
```bash
xcode-select --install            # once, if not installed (insightface compiles a small C extension)
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools cython
pip install -r requirements.txt
python download_models.py         # shows the model licence, asks for 'yes'
python app.py
```
The first time, macOS asks for camera access for your terminal (Terminal, iTerm or VS Code). Allow it. If you denied it earlier, go to **System Settings → Privacy & Security → Camera**, enable the terminal app, and restart it.

### Ubuntu / Linux
```bash
sudo apt install python3.11 python3.11-venv python3.11-dev build-essential libgl1
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools cython
pip install -r requirements.txt   # NVIDIA GPU: replace onnxruntime with onnxruntime-gpu
python download_models.py
python app.py
```
Your user needs access to `/dev/video*` (the `video` group).

## Usage
`python app.py` opens one window with three panels: **Original av.jpg | Live webcam / Captured face | Result**.

| Button | Key | Action |
|---|---|---|
| Capture Face | `C` | Grab the current webcam frame and validate the face |
| Retake | `R` | Discard the capture and go back to the live preview |
| Swap Face | `S` | Swap your face into the selected target face |
| Save Result | `W` | Save `outputs/av_face_swapped.jpg` at full resolution |
| Quit | `Q` / `Esc` | Close the app |
| – | `1`–`9` / click a box | Choose the target face when `av.jpg` contains several faces |
| – | `N` | Switch to the next detected camera |

Options:
- `--camera N`: use a specific webcam index.
- `--list-cameras`: list the detected webcams and exit.
- `--target file.jpg`: use a different target image. The default is `av.jpg`.

**Capture quality checks:** the app checks for exactly one dominant face, detection confidence ≥ 0.6, face width ≥ 100 px, head yaw and pitch within ±30°, a sharp (not blurry) image, and reasonable brightness. A rejected capture shows the reason in the status bar.

**Blending:** the 128 px swapped face is warped back into the full-resolution target. The app matches skin tone and lighting with LAB colour transfer against the original face. It then blends through a feathered mask shaped by the 106 facial landmarks and applies light sharpening to make up for the upscale.

## Model licences (read before use)
- **buffalo_l** and **inswapper_128.onnx** are InsightFace pretrained models, released for **non-commercial research purposes only**. Commercial use needs a licence from InsightFace (https://insightface.ai).
- `buffalo_l` is downloaded from the official InsightFace GitHub releases.
- InsightFace no longer hosts `inswapper_128.onnx` officially. `download_models.py` fetches it from the Hugging Face mirror `ezioruan/inswapper_128.onnx` and verifies its SHA-256 (`e4a3f08c…49e16af`).

## Responsible use
Only use images of people who have given you permission. Do not use results to deceive, impersonate or make fake identity documents.

## Troubleshooting
- **"No working webcam found":** grant camera permission (see above) and close other apps that are using the camera.
- **"No face detected in av.jpg":** the target image needs a clearly visible face.
- **Capture rejected:** follow the message (move closer, add light, face the camera, hold still).
- **insightface fails to build:** install Xcode Command Line Tools (macOS) or `build-essential python3.11-dev` (Linux).
- **Speed:** CPU is used by default, and one swap takes about 1 s. To try Apple's CoreML, run `FACESWAP_PROVIDERS=coreml python app.py`.

## Project structure
```
face-swap/
├── av.jpg               # target image (read-only)
├── app.py               # GUI + workflow (entry point)
├── webcam.py            # physical webcam discovery and capture (OpenCV)
├── face_detector.py     # InsightFace detection + capture quality checks
├── face_swapper.py      # inswapper_128 ONNX face swap
├── image_processor.py   # loading, colour matching, mask blending, saving
├── download_models.py   # licence notice + model download/verification
├── requirements.txt
├── models/              # model weights (downloaded)
└── outputs/             # results
```
# face-swap-python
