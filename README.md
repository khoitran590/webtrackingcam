# Hand Gesture Tracker

Real-time webcam hand tracking and static gesture recognition with MediaPipe Tasks and OpenCV.
It tracks up to two hands and recognizes Fist, Open Palm, Thumbs Up, Thumbs Down, Pointing,
Peace / Victory, Rock On, Spider-Man / ILY and OK Sign, at any hand rotation.

This implements `hand_gesture_tracking_specification.md`. See [SPEC_NOTES.md](SPEC_NOTES.md) for
where that spec falls short and how each gap is handled here.

## Setup

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
# or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Don't install `opencv-python` separately. `mediapipe` already installs `opencv-contrib-python`.

## Run

```bash
.venv/bin/python hand_gesture_tracker.py            # --camera 1 to pick another camera
```

The hand model (~7.8 MB) is downloaded to `models/` on first run and checked against a pinned SHA-256.
To quit, press `q` or `Esc`, or close the window.

**macOS:** allow camera access for your terminal app in System Settings → Privacy & Security → Camera,
then restart the terminal.

## Code layout

- `gesture_logic.py`: finger states, gesture classification, label voting, smoothing and hand tracking.
  It has no camera or MediaPipe dependency.
- `hand_gesture_tracker.py`: model download, the MediaPipe pipeline, drawing, and the camera loop.

## Tests

```bash
uv pip install -r requirements-dev.txt && .venv/bin/python -m pytest
```

The first run downloads the model and 10 sample hand photos into `tests/.cache/`.
