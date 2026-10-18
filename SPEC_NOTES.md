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
| C9 | No tests. | **Resolved**: 218 tests (`pytest`), plus `tools/evaluate_hagrid.py` and `tools/evaluate_faces.py`, which measure accuracy on real photos (sections D, E, F and G). |

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
| **Call Me** *(new: thumb and pinky)* | call | 83% → 89% | **74% → 92%** |
| **L Shape** *(new: thumb and index)* | thumb_index | 92% → 93% | 98% → 98% |
| Thumbs Up | like | 82% → 88% | 86% → 90% |
| Thumbs Down | dislike | 76% → 78% | 84% → 84% |
| Fist | fist | 90% → 92% | 82% → 88% |
| Open Palm, Peace / Victory, Pointing | palm, peace, one | 98/93/92% → 98/94/94% | 93/93/92% → 94/93/94% |
| **Mean of the 10** | | **88.6% → 91.4%** | **88.7% → 92.7%** |

Photo by photo, the model fixes 26 held-out gesture photos and breaks none: +3.2 points, 95% bootstrap interval
+2.1 to +4.5 (calibration set: 31 fixed, 3 broken, +2.8 points, +1.7 to +4.0). These are for the model shipped
in E9; the one before it fixed 24 and broke 4 held-out photos (+2.5).

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
| D2 | Call Me was the weakest single-hand gesture (68% held-out), mostly because the thumb isn't detected as extended. | **Resolved** by the learned model: 89–92%. |
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
| E1 | **The rules name every hand**, so a relaxed hand reads as Fist or Open Palm: 76% of HaGRID "no gesture" photos got a gesture name. | **Improved**: the model hides the label of a hand it is ≥80% sure is idle: 16–27%. |
| E2 | **Shapes the app has no name for got a wrong name** 29–32% of the time; a pen grip got one in 94–98% of photos (on training photos, mostly Thumbs Up). Falling back to the rules whenever the model said "other" kept that error: on training data, 97% of grips were still named. | **Resolved**: when the model is sure of "other", finger counts still show but gesture names don't: 10%. |
| E3 | **A model on raw landmarks fails on rotated hands**: trained on mostly upright hands, it recognised 98% of upright held-out hands but 0–2% of the same hands turned 90° or 180°. Giving it the hand's angle as an extra input brought the problem back (54% at 180°). | **Resolved**: hands are turned upright first (98% at every angle). Unit-tested. |
| E4 | **HaGRID's idle hands are far away and hanging at people's sides.** Every model trained on them called close-up, spread open palms (the sample photos) "no gesture", the larger networks with up to 99% confidence. Mirroring training hands to simulate the back of the hand, adding the hand angle, and a finger-spread rule did not fix it. | **Open**: the "no gesture" gate needs ≥80% confidence (the 64-unit model peaks at 77% on those palms; larger models were picked against for this). Recalibrate with your own takes (`tools/record_samples.py`). |
| E5 | **The live app used stricter MediaPipe settings than every measurement**: detection 0.7 vs. 0.5. At 0.7, 89.6% of held-out gesture photos had a hand found instead of 94.2%, and two-hand gestures dropped from 76% to 64%. | **Resolved**: detection and presence 0.3 for both the app and the evaluator. That finds a hand in 96.0%, both hands of a two-hand gesture in 86% (was 67%). About 0.3% of photos gain a phantom "hand"; at 0.2, faces and kettles were detected too. `--min-detection` adjusts it live. |
| E6 | **First live retrain (2026-10-17)** with 6 takes (601 hands): 3 "no gesture" and 3 Open Palm. Live, the shipped model labelled the relaxed hands correctly in 40% of frames (else Open Palm, 4 Fingers Up, Thumb Out). Retrained, each take held out: 45%, HaGRID no_gesture 64.7% -> 73.5%, but Pointing 82.9% -> 74.4%, and MediaPipe's back-of-hand Open Palm photos read "no gesture" at 82-92% (over the 0.8 gate), so they lost their label. With the gate at 0.93 the photos pass but the relaxed hands fall to 30%. | **Open**: shipped model kept. The relaxed-hand takes were mostly open hands; record curled or half-closed idle hands (typing, phone, chin on hand) and back-of-hand Open Palms, then retrain. |
| E7 | **Second live retrain (2026-10-17)** with 16 takes: 6 "no gesture", 7 Open Palm, 3 Pointing. Scored with each take held out, the retrained model got the "no gesture" takes right in 63% of frames vs. the shipped model's 66%. Per take it moved errors rather than removing them: open relaxed hands improved (14% -> 31%), but a curled resting hand the shipped model got 100% right read as Fist (56%), and another as Call Me. Open Palm and Pointing takes: 100% for both. HaGRID no_gesture 64.7% -> 73.0%, middle_finger 93.3% -> 88.1%. Back-of-hand palm photos passed, one at P(none) 0.80, right at the gate. | **Shipped model kept.** As still shapes, a relaxed open hand is close to Open Palm and a curled one to Fist; more takes swing single takes by up to 40 points. The takes stay in `samples/` for later retrains. |
| E8 | **Averaging the hand model's probabilities over time** (as faces do, F) was simulated on the 16 takes with the shipped model, before the 5-frame vote. Time constants 0.1-0.5 s: shown-label accuracy 85.0% -> 83.7-84.3% ("no gesture" takes 67.0% -> 63.9-65.4%), label changes 31 -> 21-26, and the label took longer to clear when a hand stopped gesturing (0.10 s -> 0.31 s at 0.1 s, 0.78 s at 0.3 s), because the averaged "no gesture" probability has to climb to the 0.8 gate. The model's mistakes last whole takes, so averaging doesn't fix them. | **Not adopted.** The takes are steady 4 s holds; flicker while a hand moves between shapes is unmeasured. |
| E9 | **Third live retrain (2026-10-17)**, shipped: 29 takes (2,509 hands): 6 "no gesture", 12 Open Palm, 5 Pointing, 4 Fist, 2 Call Me. The shipped model hid close-up fists: on 2 of 4 Fist takes it was ≥0.8 sure of "no gesture" (one take hidden in every frame), and on a knuckles-on fist it said Pointing, which needs only 0.4. Two guards were tried first and dropped: using the model's Pointing only when the rules see the index up broke pointing at the camera (HaGRID point 83% → 17%, your Pointing takes 100% → 68%), and never hiding a hand the rules call Fist showed curled idle hands as Fist (your "no gesture" takes 66% → 37%, HaGRID no_gesture 65% → 45%). Retrained with the takes, each take held out: Fist 45% → 87%, Call Me 53% → 77%, "no gesture" 66% → 72%, Open Palm and Pointing about 100% for both. App on HaGRID (section D): gestures 91.4% / 91.8% → 91.4% / 92.7% (calibration / held-out), no_gesture photos given a name 23% / 18% → 27% / 16%, two-hand gestures unchanged, no class moving more than 4 points. Back-of-hand palm photos: highest P(none) 0.76 → 0.68. | **Shipped.** Still weak: a knuckles-on fist (47%), and a palm-facing Call Me where MediaPipe seems to swap thumb and index landmarks (53%; the rules then see Rock On). |

## F. Facial expressions

Not in the spec. `face_expression.py` finds faces with MediaPipe's face landmarker, turns each face so its eyes
are level, crops it, and classifies it with [HSEmotion](https://github.com/sb-ai-lab/EmotiEffLib)'s
EfficientNet-B0 (`enet_b0_8_best_afew`, trained on AffectNet, Apache-2.0) through OpenCV's DNN module. This adds
no new dependency. Probabilities are averaged over time per face (0.3 s time constant), and a face gets no label
while no expression is at least 40% likely (50% before F10). The labels name the expression a face shows, not what the person feels.

**Choice of engine.** Mean per-expression accuracy (chance 14–17%), before any of the changes below:

| Engine | AffectNet (2.7k photos, 7 expressions) | RAF-DB (10.6k photos; this copy had no Neutral, so 6 expressions) | Time per face |
|---|---|---|---|
| FER+ (ONNX model zoo) | 41% | 35% | 7 ms |
| MediaPipe's 52 face blendshapes + logistic regression | 46% | 55% (trained and tested on RAF-DB itself) | ~0 ms |
| **HSEmotion EfficientNet-B0** | 66% (may include its training photos) | **61%** | 9 ms |
| HSEmotion EfficientNet-B2 | 68% | 63% | 16 ms |

**Accuracy.** [RAF-DB](http://www.whdeng.cn/RAF/model1.html)'s official split, which the network never saw:
**train** (12,271 photos) was used to choose the crop, the cut-off and the offsets (F3), and **test** (3,068)
was not used for anything. AffectNet (5,235 photos, contempt excluded) is a re-split of unknown origin that may
contain the network's own training photos, and half of it was used to fit the offsets (F4). All numbers count faces MediaPipe found (89% of RAF-DB's 100 px
photos and 99.7% of AffectNet's). Run `tools/evaluate_faces.py` to reproduce them.

| Expression | RAF-DB train | RAF-DB test | AffectNet |
|---|---|---|---|
| Happy | 87% | 86% | 87% |
| Sad | 83% | 85% | 61% |
| Neutral | 77% | 78% | 48% |
| Angry | 71% | 79% | 56% |
| Fearful | 69% | 66% | 60% |
| Surprised | 53% | 53% | 81% |
| Disgusted | 47% | 47% | 73% |
| **Mean** | **69.5%** | **70.6%** | **66.6%** |

At the 40% cut-off (F10), 92.5% of RAF-DB test faces get a label and 81% of those labels are right (at 50%: 81% and 85%; 78% with no cut-off).
The share of faces shown, and how often the shown label is right, at each cut-off is printed by the evaluator.

| # | Finding | Status |
|---|---------|--------|
| F1 | **A tilted head defeats the network**, which was trained on upright faces. On RAF-DB train photos turned 45° (300 per expression, before F3), accuracy fell from 67% to 42%; MediaPipe's own 90°-turned test portrait drops from Happy at 100% to 52%. | **Resolved**: the crop is turned so the eyes are level: 67% upright, 67% at 20° and at 45°, and no change on upright faces (67.1% vs. 67.3% without). Unit-tested at six angles, plus the real 90° portrait. |
| F2 | The network has an 8th class, **contempt**: AffectNet's least reliable class, and RAF-DB has no contempt photos to measure it. | **Resolved**: dropped, and the other seven renormalised. |
| F3 | **The network calls a resting face Angry** about one time in five (RAF-DB 20%, AffectNet 15%), and on RAF-DB it calls Surprised faces Fearful (46%) and Disgusted faces Angry (51%). | **Improved**: per-expression offsets added to the network's scores (`CLASS_OFFSETS`): RAF-DB test 65.7% → 70.6% (+4.9 points, 95% bootstrap interval +3.2 to +6.7), resting faces read as Angry 20% → 8% (AffectNet 15% → 5%), Surprised 32% → 53%. Costs: Angry recall 88% → 79% (AffectNet 72% → 57%), Fearful 74% → 66% (75% → 58%). |
| F4 | **RAF-DB and AffectNet disagree on Surprised vs. Fearful.** Offsets fitted to RAF-DB alone reached 71.9% on its test split but dropped AffectNet from 66% to 58% (Fearful 75% → 35%). | **Resolved**: offsets are fitted on RAF-DB train and half of AffectNet together (`tools/fit_expression_offsets.py`); the unseen AffectNet half stays level (65.9% → 65.7%). |
| F5 | With the offsets, **AffectNet's resting faces read as Surprised 21%** of the time (16% before), so on that dataset part of the false Angry moved to Surprised. The first live check agrees: a resting face at the webcam (5 s, 134 frames, one person) read Surprised in 17% of frames, Neutral in 28%, and nothing in the rest. | **Open**: record resting-face takes (F9) and refit. |
| F6 | **Disgusted is the weakest expression** (47% on RAF-DB; mostly read as Angry) and Surprised is still read as Fearful 27% of the time on RAF-DB. | **Open** |
| F7 | The crop margin matters: crops 1.0× the landmarks' extent scored best of 0.8–1.4 (67.1% vs. 66.0% at 1.1 and 61.6% at 1.4, on about 1,780 train faces, before F3). | **Resolved**: 1.0. |
| F8 | Speed: the face landmarker and network add about 12 ms per frame for one face (6.5 → 18.8 ms per 720p frame on an Apple M5, drawing included). | **Resolved**: one face by default (`--max-faces`); `f` or `--no-faces` turns faces off. |
| F9 | **The offsets come from other people's photos.** `tools/record_samples.py` records expression takes (letter keys; only the network's scores are kept, with the app's offsets removed, in `samples/faces/`), and `tools/fit_expression_offsets.py` adds them as a third set weighted like each dataset; each expression's takes alternate between fitting and checking. First live session (2026-10-17, one person, 11 takes: Neutral 4, Happy 3, Surprised 2, Sad 2): the current offsets already pick the right expression in 95.9% of held-out frames (resting face 90-100% Neutral; the 17% Surprised of F5 did not recur). A refit scored 93.6% on the held-out takes and cost 2.5 points on RAF-DB test. | **Resolved**: offsets kept. What fell short live was confidence, not the choice of expression (F10). |
| F10 | **At the 50% cut-off, correctly read faces got no label**: the resting face's top probability was 0.49-0.60 (no label in 7-56% of frames), Sad's 0.39-0.41 (never shown). | **Resolved**: cut-off lowered to 40%. Labelled resting-face frames 44-93% -> 87-100%; RAF-DB test: shown 80.8% -> 92.5%, right 85% -> 81%. Sad is still shown in only 20-26% of frames. |

## G. Hands and faces together

The hand and face pipelines used to run independently. Each frame now goes faces → hands → expressions:
the face boxes are used to drop phantom hands, and the confirmed hands are used to tell which faces they cover.
Measured with the app's own functions on the HaGRID calibration photos of 10 classes (mute, no_gesture, call, holy,
hand_heart, peace, one, palm, like, ok: 1,260 hands, 677 faces), on AffectNet (5,235 photos) and on RAF-DB test (3,068).

| # | Finding | Status |
|---|---------|--------|
| G1 | **Faces produce phantom hands.** At the app's 0.3 detection setting (E5), 8–13% of close-up face photos get a "hand": squashed sets of landmarks on an eyebrow, lip, nostril or hairline. Their handedness scores are as high as real hands' (median 0.8–0.93), so the score can't filter them. | **Resolved**: a hand mostly (≥50% of landmarks) inside a face box with a palm shorter than 0.3 face sides is dropped (`face_phantom`). Phantoms measured 0.02–0.3; real hands held in front of a face in HaGRID measured 0.35–0.71. This drops 56% of hand detections on AffectNet and 89% on RAF-DB, and **0 of 1,260** HaGRID hands, including hands over the mouth and Prayer. Some dropped AffectNet detections are real hands whose landmarks had collapsed, which were unusable anyway. |
| G2 | **Phantoms could stay on screen** (Not verified, below). | **Improved**: a new hand is shown only after 3 frames in a row (`HAND_CONFIRM_FRAMES`, about 0.1 s), and an unconfirmed track is dropped the first frame it is missed, so a detection that flickers in and out never appears. Confirmed hands still survive 10 missed frames. Still photos and the evaluators use 1. |
| G3 | **A hand over the face.** The face landmarker still places landmarks behind a hand, and the network classifies the crop through it. On HaGRID faces a confirmed hand covers, the network gave Sad / Surprised / Fearful / Angry 15% of the time vs. 8% on clear faces, and no label 52% vs. 24%. On AffectNet, covered faces scored 65.2% vs. 66.6% (46 faces; AffectNet's own photos often show a hand over the mouth in Surprised and Fearful, so the network may use the hand as a cue). | **Resolved**: a face is *covered* when a hand's outline (widened by 0.15 palm lengths) is over at least half of an eye's or the mouth's landmarks (`covered_regions`). A covered face shows "Face covered" and keeps its last smoothed expression, unshown, until the hand moves away. It fires on 57 of 59 finger-on-lips (mute) faces, 42 of 70 no_gesture faces (in this shard mostly hands over the mouth), and 0–4 per gesture class elsewhere (Prayer and Heart held in front of the face). The outline includes the gap between thumb and fingers, so the region list can name an eye the thumb only passes. |

## H. Keeping hands apart when they cross

There is no video with ground truth, so tracking is measured by simulation: two real HaGRID hands (random pairs from
the training landmarks) move past each other at 0.5, 1 and 2 frame-widths per second, 30 fps, with 0.004 landmark
jitter, 5% of detections dropped, the front hand hiding the other half the time while they overlap, and 5% of
Left/Right labels flipped. 102 runs per cell (3 vertical gaps × 34 seeds); a switch is a shown (confirmed) hand
changing identity.

| Shown-identity switches per 102 runs (slow / medium / fast) | Crossing | Meet and part | Side by side |
|---|---|---|---|
| Nearest-wrist matching (before) | 66 / 66 / 72 | 0 / 0 / 0 | 0 / 0 / 14 |
| + velocity prediction and best joint matching | 46 / 66 / 52 | 0 / 2 / 56 | 6 / 6 / 6 |
| **+ Left/Right label penalty 0.03 (the app)** | **8 / 6 / 10** | 0 / 0 / 2 | 6 / 6 / 6 |
| Same, penalty 0.05 | 20 / 12 / 12 | 2 / 0 / 2 | 16 / 11 / 8 |

| # | Finding | Status |
|---|---------|--------|
| H1 | **Crossing hands swapped identities** (and with them their gesture votes and smoothing) in about two of three crossings: right after they pass, each hand is nearest to where the other one was. | **Resolved**: `HandTracker` predicts each wrist from its smoothed per-frame velocity (up to 3 missed frames ahead), picks the matching with the lowest total cost, and adds 0.03 for a label that disagrees with the track's usual one. The penalty was chosen from 0.01-0.05 on the runs above and checked on other hands and seeds: with perfect labels 0 crossing swaps (was 60-78), and with 15% of labels flipped 46 / 4 / 10. Unit tests cover a crossing with and without labels and a mislabelled hand. |
| H2 | Velocity prediction alone makes hands that meet and part swap (56 of 102 at 2 widths/s), because it predicts they carry on through each other. The label penalty undoes this. | **Open, small**: with 15% of labels flipped, fast meet-and-part (like a clap) swapped 28 of 102 times on the check runs (nearest-wrist: 0). Live label noise is unmeasured. |
| H3 | When MediaPipe gives both hands the same label, the penalty applies equally to both ways of pairing them, so it cancels out; only labels that disagree with each other steer the matching. | By design. |

## Not verified

- **A live webcam run.** The terminal can open the camera. The sample recorder ran live (17 takes, about 15 fps with drawing; E6, F9, F10); the app's own window and keys have not been tried live.
- **Live Left/Right labels.** Handedness is verified on photos, raw and mirrored, but not on a live mirrored webcam feed.
- **Spider-Man / ILY on real hands.** The other gestures are measured on real photos (section D), but there is no ILY photo set with ground truth.
- **All gestures, the learned model and the "no gesture" gate on a live webcam.** Sections D and E come from still photos of distant hands.
- **Phantom hands in video.** At the lower detection settings (E5), a phantom detection could persist through tracking. Phantoms on faces are dropped (G1) and flickering ones are never shown (G2), but a phantom detected steadily for 3 frames away from any face would still be shown. One was seen live (2026-10-17): a curled "hand" about 0.7 times the size of the real one, right beside its little-finger side, in 20 of 76 frames of one Open Palm take. It did not come back in 13 one-hand takes recorded with `--save-extra-hands`, so there are no images of it yet.
- **Hand identities through crossings on live video** (H1). The simulation's label-flip rate is an assumption.
- **Smoothing settings.** The One-Euro values (`min_cutoff=1.0, beta=10`) were set by reasoning about units, not tuned on live video.
- **Facial expressions on a live webcam.** Section F comes from still photos of other people. The 0.3 s smoothing,
  the offsets (F3) and the 50% cut-off have not been checked on video, nor have glasses, side lighting or a face
  turned away from the camera.
- **Spec §6 Phases 2–3:** the ML classifier for static gestures is implemented (section E); dynamic gestures, mouse control and the WebSocket bus are not.
