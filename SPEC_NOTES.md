# Where `hand_gesture_tracking_specification.md` falls short

This is a review of the spec measured against an actual implementation on macOS (Apple Silicon) in October 2026.
Evidence comes from running the spec's own heuristics, ported unchanged to the current MediaPipe API, on
10 sample hand photos. `tests/test_sample_images.py` downloads the same photos.

Status: **Resolved** = fixed here and covered by a test or a check. **Open** = still a limitation.

## A. The spec will not run as written

| # | Shortfall | Evidence | Status |
|---|-----------|----------|--------|
| A1 | `pip install mediapipe` now installs mediapipe **1.1.0**, which has **no `mp.solutions`**. The reference code fails on the first line of `main()` with `AttributeError`. | `dir(mediapipe)` → `['Image', 'ImageFormat', 'tasks']` | **Resolved**: uses `mp.tasks.vision.HandLandmarker`. The `.task` model the Tasks API needs is downloaded on first run from a pinned URL and checked against a pinned SHA-256. |
| A2 | The **Left/Right labels are reversed** between the legacy and Tasks models. Legacy `solutions.hands` assumed a mirrored input. The Tasks model labels the hand as it *appears* in the image. With the spec's flip-then-detect pipeline, a straight port shows "Left" for your right hand and inverts the spec's thumb rule. | Same photo (`left_hands.jpg`, back of a left hand, nails visible): legacy 0.10.21 → `Right`; Tasks 1.1.0 → `Left` | **Resolved**: `user_handedness()` swaps the label for mirrored frames. Tests check it on raw and mirrored photos. With the swap removed, all 10 mirrored tests fail. |
| A3 | `pip install opencv-python mediapipe` installs **two OpenCV distributions**, because `mediapipe` already requires `opencv-contrib-python`. Both write the same `cv2/` package. | `Requires-Dist: opencv-contrib-python` in mediapipe's metadata | **Resolved**: `requirements.txt` lists only `mediapipe`. A clean install has exactly one OpenCV and `pip check` is clean. |
| A4 | The "Python 3.8–3.11" range is stale and nothing is pinned. | OpenCV is now 5.0, NumPy 2.x | **Resolved**: versions are pinned. Clean installs pass all tests on Python 3.12 and 3.14. |

## B. The gesture logic is wrong on real hands

| Photo | Truth | Spec heuristic | This implementation |
|---|---|---|---|
| fist | Fist | **Thumbs Up** | Fist |
| pointing (×2) | Pointing | **2 Fingers Up** | Pointing |
| victory (×2) | Peace / Victory | **3 Fingers Up** | Peace / Victory |
| thumbs up (×2) | Thumbs Up | **3 Fingers Up** / **Fist** | Thumbs Up |
| thumbs down | Thumbs Down | **4 Fingers Up** | Thumbs Down |
| open hand, fingers pointing down (×2) | Open Palm | **Fist** | Open Palm |
| open hand, upright (×2) | Open Palm | Open Palm | Open Palm |

Every photo gives the same result when mirrored, as the live app does.

| # | Shortfall | Status |
|---|-----------|--------|
| B1 | The thumb test `x4 < x3` compares two joints a few pixels apart along one axis. It gave false positives on the fist, pointing and victory photos. It also depends on the handedness label (A2) and on palm vs. back of hand. | **Resolved**: the thumb counts as extended when its tip is more than 0.80 palm lengths from the ring knuckle (3D world landmarks): 97% balanced accuracy on HaGRID (section D). The rule doesn't use handedness. A first version (tip to pinky knuckle > 0.88), tuned on the 10 photos, let 22% of HaGRID fists read as Thumbs Up. |
| B2 | **"Thumbs Up" contradicts the thumb rule.** The thumb points *vertically* in a thumbs-up, but the rule checks *horizontal* extension. | **Resolved**: the thumb's direction in pixel space picks Thumbs Up or Thumbs Down when it is within 60° of vertical, otherwise Thumb Out. |
| B3 | `y_tip < y_pip` only works for an upright hand. | **Resolved**: wrist→tip vs. wrist→PIP distance ratio in 3D world space, cutoff 1.25 (1.17 for the pinky). That's 97–99% balanced accuracy per finger on about 3,600 HaGRID fingers. Tested under 200 random rotations, scales and mirrorings per pose. The spec's rule fails that test for 8 of 8 poses. |
| B4 | The pinch distance uses normalized coordinates, so it is stretched by the aspect ratio and depends on distance from the camera. | **Resolved**: measured in pixels and divided by the hand's palm length in pixels, so it is free of both aspect ratio and scale. (A 3D version tried first only caught 61% of real OK signs, because MediaPipe's depth estimates are too noisy for fingertip contact.) |
| B5 | Three different pinch thresholds: 0.05, 0.045 and 0.04. | **Resolved**: one constant, `PINCH_RATIO = 0.42`, calibrated on real OK signs (92% recall, 1% false positives on other open hands). The first, uncalibrated value of 0.3 caught only 36% of them. |
| B6 | No Thumbs Down; a sideways thumb counts as "Thumbs Up"; the fallback prints "1 Fingers Up". | **Resolved** |
| B7 | Each frame is classified independently, so labels flicker. | **Resolved**: per-hand 5-frame majority vote (the shown label only changes when outvoted), One-Euro smoothing of the drawn landmarks, and nearest-wrist tracking so the two hands keep separate state. |
| B8 | *(Found during implementation.)* **Pinch distance alone can't tell a pinch from a fist.** A fist's thumb and index tips can be as close as an OK sign's. OK Sign is safe because it also needs three extended fingers. A standalone "pinch = click" (spec §6, Phase 3) would fire on fists. | **Open**: there is no standalone pinch indicator. `HandReading.pinch` exposes the raw value for later work. |

## C. Runtime / UX problems

| # | Shortfall | Status |
|---|-----------|--------|
| C1 | The label is drawn at `wrist_y + 30`, which is usually below the frame. | **Resolved**: the label goes above the hand, or below it if there's no room. It's clamped on-screen, kept out of the HUD strip, and moved aside when another hand's label is already there (two-hand gestures put both hands in the same place). |
| C2 | A single failed `cap.read()` exits the app. | **Resolved**: up to 60 consecutive dropped frames are tolerated, then the app exits with a clear error. |
| C3 | Closing the window doesn't stop the loop. Also, on macOS, `getWindowProperty` **raises `cv2.error`** once the window is gone instead of returning 0, so the obvious fix would crash. | **Resolved**: both cases are handled. Checked with `destroyWindow`; an actual click on the close button hasn't been tested. |
| C4 | Nothing is released on an exception or Ctrl+C. | **Resolved**: `try/finally`, and the landmarker runs as a context manager. |
| C5 | 720p is recommended but never requested. | **Resolved**: 1280×720 by default (`--width/--height`), and the actual resolution is printed. |
| C6 | On macOS, a missing camera permission shows up as a bare "Could not access the webcam". | **Resolved**: the error explains where to grant access. Confirmed on this machine, where the terminal has no permission yet. |
| C7 | Camera index 0 is hard-coded. | **Resolved**: `--camera N`. |
| C8 | Tasks `VIDEO` mode needs strictly increasing timestamps. | **Resolved** |
| C9 | No tests. | **Resolved**: 171 tests (`pytest`), plus `tools/evaluate_hagrid.py`, which measures accuracy on real photos (sections D and E). |

## D. Gestures added after the spec, and accuracy on real photos

The spec lists OK Sign, Rock On and Spider-Man / ILY, but nothing had been checked on real hands, and OK Sign
turned out to work on only 36% of real photos. All gestures are now measured on photos from the
[HaGRID v2](https://github.com/hukenovs/hagrid) test split:

- **Calibration set:** 100 per class, used to set the rules' thresholds.
- **Held-out set:** 50–100 per class, from other shards, never used for tuning.

The learned model (section E) is trained on HaGRID's *train* split, which shows different people, so both
sets are held out for it. Run `tools/evaluate_hagrid.py` to reproduce both columns. All numbers use the app's
MediaPipe settings (E5).

**Single-hand gestures** (hit = a detected hand shows the gesture; HaGRID photos often include a second, resting hand).
Rules alone → the app (rules + model):

| Gesture | HaGRID class | Calibration | Held-out |
|---|---|---|---|
| OK Sign | ok | 92% → 98% | 92% → 98% |
| Rock On / Spider-Man / ILY | rock (HaGRID mixes thumb in and out) | 88% → 90% | 93% → 96% |
| **Call Me** *(new: thumb and pinky)* | call | 83% → 90% | **74% → 90%** |
| **L Shape** *(new: thumb and index)* | thumb_index | 92% → 94% | 98% → 98% |
| Thumbs Up | like | 82% → 89% | 86% → 90% |
| Thumbs Down | dislike | 76% → 78% | **84% → 82%** |
| Fist | fist | 90% → 92% | 82% → 86% |
| Open Palm, Peace / Victory, Pointing | palm, peace, one | 98/93/92% → 97/93/93% | 93/93/92% → 94/91/93% |
| **Mean of the 10** | | **88.6% → 91.4%** | **88.7% → 91.8%** |

Photo by photo, the model fixes 24 held-out gesture photos and breaks 4: +2.5 points, 95% bootstrap interval
+1.3 to +3.9 (calibration set: 33 fixed, 5 broken, +2.8 points, +1.7 to +4.0).

**Two-hand gestures** (new). Shown as a "Both hands: …" banner and majority-voted like single-hand labels.
The vote restarts whenever the pair of hands changes. The rules decide these alone. The first number counts every
photo; the one in brackets counts only photos where MediaPipe found both hands:

| Gesture | Rule | HaGRID class | Calibration | Held-out |
|---|---|---|---|---|
| **Heart** | thumb tips touch, index tips touch above them, wrists apart | hand_heart, hand_heart2 | 90% / 98% (90% / 100%) | 100% / 98% (100% / 98%) |
| **Frame** | two L shapes, one rotated 180°, at opposite corners | take_picture | 86% (92%) | **74% (82%)** |
| **Time Out** | one flat hand level, the other upright with fingertips at its palm | timeout | 80% (96%) | 81% (93%) |
| **X Sign** | crossed forearms: both hands lean up and away from each other | xsign | 69% (96%) | 67% (89%) |
| **Prayer** | flat hands pressed together, pointing up | holy | **44%** (98%) | **52%** (100%) |
| **Double ⟨gesture⟩** | both hands show the same named gesture (e.g. Double Spider-Man / ILY) | thumb_index2 → Double L Shape | 87% (94%) | 88% (95%) |

Two-hand false positives on all other classes: **0 of 3,400** calibration photos and **1 of 2,447** held-out
photos (a Heart fired on a Frame photo).

| # | Finding | Status |
|---|---------|--------|
| D1 | **Prayer is mostly limited by detection.** Palms pressed together often look like one hand to MediaPipe. Lower detection thresholds (E5) raised Prayer on held-out photos from 26% (the app's old settings; 36% at the 0.5 used for earlier measurements) to 52%; the rule is right 98–100% of the time once both hands are found. | **Open**: needs a different detector or a one-hand fallback. |
| D2 | Call Me was the weakest single-hand gesture (68% held-out), mostly because the thumb isn't detected as extended. | **Resolved** by the learned model: 90%. |
| D3 | **Rock On vs. Spider-Man / ILY differ only by the thumb**, and HaGRID's "rock" photos include both. The split between them can't be scored against ground truth. The model has one "rock" class; the rules' thumb state picks the name. | **Open**: no ILY photos to validate against. |
| D4 | *(Found on a real photo.)* An upright and an upside-down open hand next to each other read as **Frame**. | **Resolved**: Frame excludes hands with middle, ring and pinky all extended. Covered by a unit test and a real-photo test. |
| D5 | *(Found when rendering.)* With two hands close together, as in every two-hand gesture, the two hand labels were drawn on top of each other. | **Resolved**: see C1. |
| D6 | HaGRID shows people at a distance, so hands are small (about 50 px). A webcam close-up should give better landmarks, but these numbers haven't been checked live. | **Open**: see E4 and below. |
| D7 | **Thumbs Down is now the weakest gesture** (78–82%). Of 150 photos, 12 had no hand detected and the other 19 misses scatter (Pointing 4, no label 3, Fist 2, Thumb Out 2, 1 Finger Up 2, …). Correct thumbs are within about 55° of vertical (99th percentile); 5 misses were at 60–70°, so widening the 60° cone would fix about 2 photos but turn real sideways thumbs (Thumb Out) into up/down, which HaGRID has no photos to measure. | **Open**: check with live samples. |

## E. The learned model (spec §6 Phase 2, "ML classifier")

`gesture_model.py` adds a small neural network (one hidden layer of 64 units, 155 KB of JSON weights, numpy only)
that works alongside the rules. It is trained by `tools/train_gesture_model.py` on 9,862 hands from 10,447 photos of
HaGRID's train split: the 10 gestures, 7 variants the app shows under the same names (e.g. `stop` = Open Palm with
fingers together, `point` = Pointing at the camera), "no gesture", and 9 shapes the app has no name for.
Thresholds are tuned on cross-validation that holds out whole shards, never on the test photos.

Each hand is moved, scaled, **turned upright** and mirrored to look like a right hand before the model sees it.
The model's answer is used only above a per-class confidence; otherwise the rules decide. The rules keep thumb
direction, thumb in/out (Rock On vs. ILY), finger counts and all two-hand gestures.

| # | Finding | Status |
|---|---------|--------|
| E1 | **The rules name every hand**, so a relaxed hand reads as Fist or Open Palm: 76% of HaGRID "no gesture" photos got a gesture name. | **Improved**: the model hides the label of a hand it is ≥80% sure is idle: 18–23%. |
| E2 | **Shapes the app has no name for got a wrong name** 29–32% of the time; a pen grip got one in 94–98% of photos (on training photos, mostly Thumbs Up). Falling back to the rules whenever the model said "other" kept that error: on training data, 97% of grips were still named. | **Resolved**: when the model is sure of "other", finger counts still show but gesture names don't: 10%. |
| E3 | **A model on raw landmarks fails on rotated hands**: trained on mostly upright hands, it recognised 98% of upright held-out hands but 0–2% of the same hands turned 90° or 180°. Giving it the hand's angle as an extra input brought the problem back (54% at 180°). | **Resolved**: hands are turned upright first (98% at every angle). Unit-tested. |
| E4 | **HaGRID's idle hands are far away and hanging at people's sides.** Every model trained on them called close-up, spread open palms (the sample photos) "no gesture", the larger networks with up to 99% confidence. Mirroring training hands to simulate the back of the hand, adding the hand angle, and a finger-spread rule did not fix it. | **Open**: the "no gesture" gate needs ≥80% confidence (the 64-unit model peaks at 77% on those palms; larger models were picked against for this). Recalibrate with your own takes (`tools/record_samples.py`). |
| E5 | **The live app used stricter MediaPipe settings than every measurement**: detection 0.7 vs. 0.5. At 0.7, 89.6% of held-out gesture photos had a hand found instead of 94.2%, and two-hand gestures dropped from 76% to 64%. | **Resolved**: detection and presence 0.3 for both the app and the evaluator. That finds a hand in 96.0%, both hands of a two-hand gesture in 86% (was 67%). About 0.3% of photos gain a phantom "hand"; at 0.2, faces and kettles were detected too. `--min-detection` adjusts it live. |

## Not verified

- **A live webcam run.** This terminal has no camera permission, so the loop and the sample recorder were tested with a fake camera that replays photos.
- **Live Left/Right labels.** Handedness is verified on photos, raw and mirrored, but not on a live mirrored webcam feed.
- **Spider-Man / ILY on real hands.** The other gestures are measured on real photos (section D), but there is no ILY photo set with ground truth.
- **All gestures, the learned model and the "no gesture" gate on a live webcam.** Sections D and E come from still photos of distant hands.
- **Phantom hands in video.** At the lower detection settings (E5), a phantom detection could persist through tracking; photos cannot show this.
- **Smoothing settings.** The One-Euro values (`min_cutoff=1.0, beta=10`) were set by reasoning about units, not tuned on live video.
- **Spec §6 Phases 2–3:** the ML classifier for static gestures is implemented (section E); dynamic gestures, mouse control and the WebSocket bus are not.
