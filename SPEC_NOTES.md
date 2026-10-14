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
| B1 | The thumb test `x4 < x3` compares two joints a few pixels apart along one axis. It gave false positives on the fist, pointing and victory photos. It also depends on the handedness label (A2) and on palm vs. back of hand. | **Resolved**: the thumb counts as extended when its tip is more than 0.88 palm lengths from the pinky knuckle (3D world landmarks). Measured: extended ≥ 0.98, tucked ≤ 0.78. The rule doesn't use handedness. |
| B2 | **"Thumbs Up" contradicts the thumb rule.** The thumb points *vertically* in a thumbs-up, but the rule checks *horizontal* extension. | **Resolved**: the thumb's direction in pixel space picks Thumbs Up, Thumbs Down or Thumb Out. |
| B3 | `y_tip < y_pip` only works for an upright hand. | **Resolved**: wrist→tip vs. wrist→PIP distance ratio in 3D world space, cutoff 1.25 (measured: extended ≥ 1.33, curled ≤ 1.16 across 96 fingers). Tested under 200 random rotations, scales and mirrorings per pose. The spec's rule fails that test for 8 of 8 poses. |
| B4 | The pinch distance uses normalized coordinates, so it is stretched by the aspect ratio and depends on distance from the camera. | **Resolved**: measured in palm lengths on metric world landmarks. |
| B5 | Three different pinch thresholds: 0.05, 0.045 and 0.04. | **Resolved**: one constant, `PINCH_RATIO = 0.3`. **Not calibrated on real OK-sign photos.** |
| B6 | No Thumbs Down; a sideways thumb counts as "Thumbs Up"; the fallback prints "1 Fingers Up". | **Resolved** |
| B7 | Each frame is classified independently, so labels flicker. | **Resolved**: per-hand 5-frame majority vote (the shown label only changes when outvoted), One-Euro smoothing of the drawn landmarks, and nearest-wrist tracking so the two hands keep separate state. |
| B8 | *(Found during implementation.)* **Pinch distance alone can't tell a pinch from a fist.** A mirrored fist measured 0.175 palm lengths, inside the range for touching tips. OK Sign is safe because it also needs three extended fingers. A standalone "pinch = click" (spec §6, Phase 3) would fire on fists. | **Open**: there is no standalone pinch indicator. `HandReading.pinch` exposes the raw value for later work. |

## C. Runtime / UX problems

| # | Shortfall | Status |
|---|-----------|--------|
| C1 | The label is drawn at `wrist_y + 30`, which is usually below the frame. | **Resolved**: the label goes above the hand, or below it if there's no room. It's clamped on-screen and kept out of the HUD strip. |
| C2 | A single failed `cap.read()` exits the app. | **Resolved**: up to 60 consecutive dropped frames are tolerated, then the app exits with a clear error. |
| C3 | Closing the window doesn't stop the loop. Also, on macOS, `getWindowProperty` **raises `cv2.error`** once the window is gone instead of returning 0, so the obvious fix would crash. | **Resolved**: both cases are handled. Checked with `destroyWindow`; an actual click on the close button hasn't been tested. |
| C4 | Nothing is released on an exception or Ctrl+C. | **Resolved**: `try/finally`, and the landmarker runs as a context manager. |
| C5 | 720p is recommended but never requested. | **Resolved**: 1280×720 by default (`--width/--height`), and the actual resolution is printed. |
| C6 | On macOS, a missing camera permission shows up as a bare "Could not access the webcam". | **Resolved**: the error explains where to grant access. Confirmed on this machine, where the terminal has no permission yet. |
| C7 | Camera index 0 is hard-coded. | **Resolved**: `--camera N`. |
| C8 | Tasks `VIDEO` mode needs strictly increasing timestamps. | **Resolved** |
| C9 | No tests. | **Resolved**: 64 tests (`pytest`). |

## Not verified

- **A live webcam run.** This terminal has no camera permission, so the loop was tested with a fake camera that replays photos.
- **Live Left/Right labels.** Handedness is verified on photos, raw and mirrored, but not on a live mirrored webcam feed.
- **OK Sign, Rock On and Spider-Man / ILY on real hands.** These are tested only on synthetic landmarks; the sample photos don't include them.
- **Smoothing settings.** The One-Euro values (`min_cutoff=1.0, beta=10`) were set by reasoning about units, not tuned on live video.
- **Spec §6 Phases 2–3** (ML classifier, dynamic gestures, mouse control, WebSocket bus) are not implemented.
