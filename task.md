# Project: Live Webcam Face Swap with Target Image

Build a Python application that replaces the face in an existing image named `av.jpg` with my face captured directly from my laptop's webcam.

## Main Objective

The application must use:

- Target image: `av.jpg`
- Source face: Live webcam feed
- Output: Face-swapped version of `av.jpg`

The webcam should capture my face in real time, and the application should use that captured face to replace the face detected in `av.jpg`.

## Technology Stack

- Python 3.11
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy
- Pillow
- Gradio or OpenCV GUI

## Application Workflow

### Step 1: Load Target Image

- Automatically load `av.jpg` from the project root.
- Display the original image.
- Detect all faces in the target image.
- If multiple faces exist, allow manual selection of the target face.

### Step 2: Open System Webcam

- Detect available webcams.
- Open the laptop's physical webcam using OpenCV.
- Display a live camera preview.
- Allow the user to capture their face by pressing the Capture button.
- Provide an option to retake the photo.
- Detect the captured face and validate its quality.

### Step 3: Face Swap

- Use the captured webcam image as the source face.
- Use `av.jpg` as the target image.
- Perform face detection and alignment.
- Use a compatible ONNX face-swap model.
- Replace only the selected face.
- Preserve the original target image's:
  - Body
  - Clothes
  - Pose
  - Background
  - Lighting and composition

### Step 4: Face Blending

- Apply natural skin-tone matching.
- Correct face alignment.
- Blend face boundaries smoothly.
- Preserve target image resolution.
- Avoid visible artifacts and unnatural edges.

### Step 5: Preview and Save

Display:

- Original `av.jpg`
- Captured webcam face
- Final face-swapped image

Provide:

- Capture Face button
- Retake button
- Swap Face button
- Save Result button

Save the final result as:

`outputs/av_face_swapped.jpg`

## Project Structure

face-swap/
├── av.jpg
├── app.py
├── webcam.py
├── face_detector.py
├── face_swapper.py
├── image_processor.py
├── requirements.txt
├── README.md
├── models/
├── outputs/

## Important Requirements

1. Use the actual physical laptop webcam.
2. Do not use prerecorded videos or simulated webcam feeds.
3. Do not use a static placeholder image as the webcam source.
4. Validate that exactly the intended target face is selected.
5. Handle cases where no face is detected.
6. Support macOS Apple Silicon and Ubuntu/Linux.
7. Check model licensing before downloading or using model weights.
8. Never overwrite the original `av.jpg`.
9. Use images only with the necessary permission.

## Execution

First inspect my Python environment and verify that `av.jpg` exists.

Then implement the complete application.

Provide a virtual environment setup and requirements.txt.

The final application should run using:

`python app.py`

## Expected Result

When I execute the application:

1. It opens my laptop webcam.
2. I see my live face in the preview.
3. I capture my face.
4. The application replaces the face in `av.jpg` with my captured face.
5. I can preview and save the final edited image.
