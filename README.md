# Hand Gesture and Face Expression Tracker

Real-time webcam hand tracking, static gesture recognition and facial expression recognition with
MediaPipe Tasks and OpenCV. It tracks up to two hands and works at any hand rotation and head tilt.

- **One hand:** Fist, Open Palm, Thumbs Up, Thumbs Down, Thumb Out, Pointing, Peace / Victory,
  Rock On, Spider-Man / ILY, OK Sign, Call Me, L Shape. Anything else is labeled by finger count
  ("3 Fingers Up").
- **Both hands** (shown as a banner): Heart, Frame, Time Out, X Sign (crossed arms), Prayer, and
  "Double ⟨gesture⟩" when both hands show the same one-hand gesture (e.g. Double Spider-Man / ILY).
- **Rules + a small learned model.** The rules handle rotation, thumb direction and two-hand gestures; the
  model (trained on HaGRID photos) is more accurate on hard shapes and recognises a hand that is *not* making
  a gesture, whose label is then hidden. Press `m` to compare with the rules alone.
- **Facial expressions:** Neutral, Happy, Sad, Surprised, Angry, Disgusted, Fearful, shown next to a box
  around the face. A face gets no label while no expression is at least 40% likely. The labels name what a
  face *shows*, not what someone feels. Press `f` to switch them off.

Accuracy on real photos is in [SPEC_NOTES.md](SPEC_NOTES.md#d-gestures-added-after-the-spec-and-accuracy-on-real-photos)
(hands) and [section F](SPEC_NOTES.md#f-facial-expressions) (faces).

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

Keys: `m` switches the learned gesture model on/off, `f` switches facial expressions on/off, `q` or `Esc`
quits (as does closing the window).
Options: `--rules-only` starts with the gesture model off; `--min-detection 0.5` makes hand detection stricter if
phantom hands appear (default 0.3); `--no-faces` skips facial expressions entirely (the face models are not
loaded); `--max-faces 2` reads more than one face (each adds about 12 ms per frame);
`--min-expression-confidence 0.7` shows fewer but more reliable expressions (default 0.4).

Three models are downloaded to `models/` on first run and checked against pinned SHA-256 sums: the hand model
(~7.8 MB), the face landmark model (~3.8 MB) and the expression network (~16 MB,
[HSEmotion](https://github.com/sb-ai-lab/EmotiEffLib) EfficientNet-B0, Apache-2.0).
The gesture model (`models/gesture_classifier.json`, 155 KB) is part of the repository.

**macOS:** allow camera access for your terminal app in System Settings → Privacy & Security → Camera,
then restart the terminal.

## Code layout

- `gesture_logic.py`: finger states, one- and two-hand gesture classification, label voting,
  smoothing and hand tracking. It has no camera or MediaPipe dependency.
- `gesture_model.py`: the learned classifier (numpy only) and how its answer is combined with the rules.
- `face_expression.py`: the face crop (turned so the eyes are level), the expression network, smoothing
  over time, and the face pipeline.
- `hand_gesture_tracker.py`: model downloads, the hand pipeline, drawing, and the camera loop.
- `tools/evaluate_hagrid.py`, `tools/evaluate_faces.py`: measure accuracy on real photos (see below).
- `tools/fit_expression_offsets.py`: refits the expression network's per-class offsets, optionally on your own face.
- `tools/train_gesture_model.py`, `tools/record_samples.py`: retrain the model, optionally on your own hands;
  the recorder also records your face for the offsets.
- `tools/hagrid_data.py`: reads HaGRID photos from Hugging Face.

## Tests

```bash
uv pip install -r requirements-dev.txt && .venv/bin/python -m pytest
```

The first run downloads the models, 10 sample hand photos and 2 face photos into `models/` and `tests/.cache/`.

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

Facial expressions are measured on [RAF-DB](http://www.whdeng.cn/RAF/model1.html) and AffectNet:

```bash
.venv/bin/python tools/evaluate_faces.py                      # RAF-DB train: photos the settings were chosen on
.venv/bin/python tools/evaluate_faces.py --split test         # RAF-DB test: held-out photos
.venv/bin/python tools/evaluate_faces.py --dataset affectnet  # may overlap the network's training photos
```

This downloads 33 MB (RAF-DB) or 108 MB (AffectNet) of face photos from unofficial copies on Hugging Face into
`tools/.cache/faces/`. Both datasets are licensed for non-commercial research only.

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

The same window records facial expressions (letter keys, shown on the right; `n` is your resting face, the
one that matters most). Face takes go to `samples/faces/` and keep only the network's scores, no images.
Refit the expression offsets with them, then copy the printed `CLASS_OFFSETS` into `face_expression.py`
if the check rows (your held-out takes and the public test sets) improve:

```bash
uv pip install -r requirements-eval.txt
C=tools/.cache/faces   # the network's scores on the public photos, saved once:
.venv/bin/python tools/evaluate_faces.py --no-offsets --save $C/rafdb_train.npz
.venv/bin/python tools/evaluate_faces.py --no-offsets --split test --save $C/rafdb_test.npz
.venv/bin/python tools/evaluate_faces.py --no-offsets --dataset affectnet --save $C/affectnet_val.npz
.venv/bin/python tools/fit_expression_offsets.py
```
