# Hand Gesture Tracker

Real-time webcam hand tracking and static gesture recognition with MediaPipe Tasks and OpenCV.
It tracks up to two hands and works at any hand rotation.

- **One hand:** Fist, Open Palm, Thumbs Up, Thumbs Down, Thumb Out, Pointing, Peace / Victory,
  Rock On, Spider-Man / ILY, OK Sign, Call Me, L Shape. Anything else is labeled by finger count
  ("3 Fingers Up").
- **Both hands** (shown as a banner): Heart, Frame, Time Out, X Sign (crossed arms), Prayer, and
  "Double ⟨gesture⟩" when both hands show the same one-hand gesture (e.g. Double Spider-Man / ILY).
- **Rules + a small learned model.** The rules handle rotation, thumb direction and two-hand gestures; the
  model (trained on HaGRID photos) is more accurate on hard shapes and recognises a hand that is *not* making
  a gesture, whose label is then hidden. Press `m` to compare with the rules alone.

Accuracy on real photos is in [SPEC_NOTES.md](SPEC_NOTES.md#d-gestures-added-after-the-spec-and-accuracy-on-real-photos).

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

Keys: `m` switches the learned model on/off, `q` or `Esc` quits (as does closing the window).
Options: `--rules-only` starts with the model off; `--min-detection 0.5` makes hand detection stricter if
phantom hands appear (default 0.3).

The hand model (~7.8 MB) is downloaded to `models/` on first run and checked against a pinned SHA-256.
The gesture model (`models/gesture_classifier.json`, 155 KB) is part of the repository.

**macOS:** allow camera access for your terminal app in System Settings → Privacy & Security → Camera,
then restart the terminal.

## Code layout

- `gesture_logic.py`: finger states, one- and two-hand gesture classification, label voting,
  smoothing and hand tracking. It has no camera or MediaPipe dependency.
- `gesture_model.py`: the learned classifier (numpy only) and how its answer is combined with the rules.
- `hand_gesture_tracker.py`: model download, the MediaPipe pipeline, drawing, and the camera loop.
- `tools/evaluate_hagrid.py`: measures accuracy on real photos (see below).
- `tools/train_gesture_model.py`, `tools/record_samples.py`: retrain the model, optionally on your own hands.
- `tools/hagrid_data.py`: reads HaGRID photos from Hugging Face.

## Tests

```bash
uv pip install -r requirements-dev.txt && .venv/bin/python -m pytest
```

The first run downloads the model and 10 sample hand photos into `tests/.cache/`.

## Accuracy on real photos

```bash
uv pip install -r requirements-eval.txt
.venv/bin/python tools/evaluate_hagrid.py                   # photos the thresholds were tuned on
.venv/bin/python tools/evaluate_hagrid.py --shard-offset 3  # held-out photos
```

This downloads 50–100 photos per class from the [HaGRID v2](https://github.com/hukenovs/hagrid)
test split (about 400 MB per run, cached in `tools/.cache/`). Each photo goes through the app's own
pipeline, and the script prints, for the rules alone and for rules + model: per-gesture accuracy,
how often relaxed hands and unnamed shapes get a gesture name, and two-hand false positives.

## Training the model on your own hands

The shipped model learned from 10,447 photos of other people, mostly far from the camera. Your own takes
teach it your hands, camera and light, and are the way to calibrate when a hand counts as "not a gesture":

```bash
.venv/bin/python tools/record_samples.py      # press a class key, hold the gesture, move it slowly
uv pip install -r requirements-train.txt
.venv/bin/python tools/train_gesture_model.py # downloads ~1.1 GB of HaGRID photos on first run (landmarks only are kept)
```

Takes are saved in `samples/` (not committed). Training reports accuracy on your own takes separately.
Record "no gesture" (your hands doing ordinary things) and Open Palm first; with them,
`--min-none-confidence` can be lowered from its cautious default of 0.8.
