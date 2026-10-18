"""Real-time webcam hand gestures and facial expressions (MediaPipe Tasks + OpenCV).

Usage:  python hand_gesture_tracker.py [--camera 0] [--width 1280] [--height 720] [--rules-only] [--no-faces]
Keys:   m switches the learned gesture model on/off (rules only); f switches facial expressions on/off;
        q or Esc quits, as does closing the window.
"""

import argparse
import hashlib
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import certifi
import cv2
import mediapipe as mp
import numpy as np

from gesture_logic import (
    WRIST,
    GestureVote,
    HandPose,
    HandTracker,
    classify_gesture,
    classify_two_hands,
    finger_states,
    pinch_ratio,
    thumb_direction,
)
from face_expression import (
    EXPRESSION_MODEL_SHA256,
    EXPRESSION_MODEL_URL,
    FACE_LANDMARKER_SHA256,
    FACE_LANDMARKER_URL,
    ExpressionModel,
    FacePipeline,
    create_face_landmarker,
    face_phantom,
)
from face_expression import (
    MIN_CONFIDENCE as MIN_EXPRESSION_CONFIDENCE,
)
from gesture_model import DEFAULT_CLASSIFIER_PATH, GestureModel, combine, hand_features

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
FACE_LANDMARKER_PATH = DEFAULT_MODEL_PATH.with_name("face_landmarker.task")
EXPRESSION_MODEL_PATH = DEFAULT_MODEL_PATH.with_name("expression_enet_b0_8.onnx")

WINDOW_NAME = "Hand Gesture and Face Expression Tracker"
HAND_CONNECTIONS = [(c.start, c.end) for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS]
FINGERTIPS = {4, 8, 12, 16, 20}
HUD_HEIGHT = 85  # top strip (status line + two-hand banner) kept free of hand labels
MAX_CONSECUTIVE_READ_FAILURES = 60  # ~2 s at 30 fps; cameras often drop frames while warming up
# MediaPipe confidence needed to report a hand. On held-out HaGRID photos, 0.3/0.3 instead of
# 0.7/0.5 found a hand in 96.0% of gesture photos instead of 89.6%, both hands of a two-hand
# gesture in 86% instead of 67% (Prayer 26% -> 52%), with no extra two-hand false alarms. About
# 0.3% of photos gained a phantom "hand" (a carpet, a dark sleeve); at 0.2 that grew to faces
# and kettles. Still photos cannot show whether a phantom would persist through video tracking.
MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3
# Frames in a row a new hand must be seen before it is shown (about 0.1 s at 30 fps). The low
# confidences above let in the odd phantom; a phantom that flickers in and out never gets shown.
HAND_CONFIRM_FRAMES = 3

GREEN, WHITE, BLACK, ORANGE, MAGENTA, CYAN = (0, 255, 0), (255, 255, 255), (0, 0, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path=DEFAULT_MODEL_PATH, url=MODEL_URL, sha256=MODEL_SHA256, name="hand landmark model"):
    """Download a pinned model on first use and verify its checksum."""
    path = Path(path)
    if path.exists() and _sha256(path) == sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"Downloading {name} to {path} ...")
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context, timeout=60) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the {name} ({e}). Download it manually from\n  {url}\nto\n  {path}") from e
    if _sha256(tmp) != sha256:
        tmp.unlink()
        raise RuntimeError(f"Downloaded {name} failed its SHA-256 check; refusing to use it.")
    tmp.replace(path)
    return path


def ensure_face_models():
    """(face landmarker, expression network) paths, downloaded on first use."""
    return (ensure_model(FACE_LANDMARKER_PATH, FACE_LANDMARKER_URL, FACE_LANDMARKER_SHA256, "face landmark model"),
            ensure_model(EXPRESSION_MODEL_PATH, EXPRESSION_MODEL_URL, EXPRESSION_MODEL_SHA256, "expression model"))


def create_landmarker(model_path, num_hands=2, min_detection=MIN_HAND_DETECTION_CONFIDENCE,
                      min_presence=MIN_HAND_PRESENCE_CONFIDENCE):
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection,
        min_hand_presence_confidence=min_presence,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def user_handedness(model_label, mirrored):
    """The Tasks model labels a hand by how it looks in the image it was given. A mirrored
    frame makes the user's right hand look like a left hand, so swap the label back.
    (The legacy mp.solutions model did the opposite: it assumed a mirrored input.)"""
    if not mirrored:
        return model_label
    return {"Left": "Right", "Right": "Left"}.get(model_label, model_label)


@dataclass
class HandReading:
    track_id: int
    handedness: str  # the user's actual hand
    gesture: str  # majority-voted label shown on screen; None = not making a gesture
    raw_gesture: str  # this frame's label
    fingers: list  # [thumb, index, middle, ring, pinky]
    pinch: float  # thumb-to-index tip distance in image palm lengths
    points_px: list  # 21 smoothed (x, y) image points in pixels
    rule_gesture: str = None  # this frame's label from the rules alone
    model_guess: tuple = None  # (class, confidence) from the learned model, when it is in use
    world_points: list = None  # MediaPipe's 21 world points (metres)
    image_label: str = None  # MediaPipe's handedness for this (possibly mirrored) frame
    label_score: float = None  # MediaPipe's confidence in image_label


@dataclass
class FrameReading:
    hands: list  # one HandReading per detected hand
    two_hand_gesture: str = None  # majority-voted gesture made with both hands, if any


class GesturePipeline:
    """Runs the landmarker on one frame and turns its output into per-hand readings.

    With a GestureModel, each hand's label combines the rules with the model (see
    gesture_model.combine); ``use_model`` switches back to the rules alone. A hand is only
    reported once it has been seen in ``confirm_frames`` frames in a row (1 for still photos)."""

    def __init__(self, landmarker, mirrored=True, tracker=None, model=None, confirm_frames=1):
        self.landmarker = landmarker
        self.mirrored = mirrored
        self.tracker = tracker or HandTracker(confirm_frames=confirm_frames)
        self.model = model
        self.use_model = model is not None
        self._last_timestamp_ms = -1
        self._pair = None  # track ids of the two hands the pair vote belongs to
        self._pair_vote = None

    def process(self, frame_bgr, t_seconds, faces=()):
        """``faces``: FaceBoxes found in this frame; hands that are phantoms on them are dropped."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # VIDEO mode rejects timestamps that do not strictly increase.
        timestamp_ms = max(int(t_seconds * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

        image_hands = [[(p.x, p.y, p.z) for p in hand] for hand in result.hand_landmarks]
        world_hands = [[(p.x, p.y, p.z) for p in hand] for hand in result.hand_world_landmarks]
        labels = [categories[0].category_name for categories in result.handedness]
        scores = [categories[0].score for categories in result.handedness]
        if faces:
            real = [i for i, hand in enumerate(image_hands) if not face_phantom([(x * w, y * h) for x, y, _ in hand], faces)]
            image_hands, world_hands, labels, scores = ([items[i] for i in real] for items in (image_hands, world_hands, labels, scores))
        tracks = self.tracker.update([hand[WRIST][:2] for hand in image_hands], labels)

        readings = []
        for track, image_pts, world_pts, label, score in zip(tracks, image_hands, world_hands, labels, scores, strict=True):
            points_px = [(x * w, y * h) for x, y, _ in track.smoother(image_pts, t_seconds)]
            fingers = finger_states(world_pts)
            pinch = pinch_ratio(points_px)
            thumb_dir = thumb_direction(points_px)
            rule = raw = classify_gesture(fingers, pinch, thumb_dir)
            guess = None
            if self.model is not None and self.use_model:
                # The model needs the label as MediaPipe saw it (what the hand looks like in this image).
                guess = self.model.predict(hand_features(points_px, world_pts, label))
                raw = combine(rule, fingers, thumb_dir, *guess, self.model.thresholds)
            gesture = track.vote.update(raw)
            if not self.tracker.confirmed(track):
                continue
            readings.append(HandReading(
                track_id=track.id,
                handedness=user_handedness(label, self.mirrored),
                gesture=gesture,
                raw_gesture=raw,
                fingers=fingers,
                pinch=pinch,
                points_px=points_px,
                rule_gesture=rule,
                model_guess=guess,
                world_points=world_pts,
                image_label=label,
                label_score=score,
            ))
        return FrameReading(readings, self._two_hand_gesture(readings))

    def _two_hand_gesture(self, readings):
        if len(readings) != 2:
            self._pair = None
            return None
        pair = {r.track_id for r in readings}
        if pair != self._pair:  # different hands: start a fresh vote
            self._pair, self._pair_vote = pair, GestureVote(self.tracker.vote_size)
        a, b = (HandPose(r.points_px, r.fingers, r.raw_gesture) for r in readings)
        return self._pair_vote.update(classify_two_hands(a, b))


def draw_text(frame, text, origin, color=GREEN, scale=0.8):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 5, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_hand(frame, reading, placed=None):
    """Draw one hand and its label. Pass the same ``placed`` list for every hand in a
    frame so labels of hands that are close together do not overlap."""
    h, w = frame.shape[:2]
    pts = [(int(round(x)), int(round(y))) for x, y in reading.points_px]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], WHITE, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 6 if i in FINGERTIPS else 4, ORANGE if i in FINGERTIPS else GREEN, -1, cv2.LINE_AA)

    # A hand that is not making a gesture keeps its skeleton and Left/Right, without a name.
    text = f"{reading.handedness}: {reading.gesture}" if reading.gesture else reading.handedness
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    origin = label_origin(pts, (tw, th), (w, h), top=HUD_HEIGHT)
    if placed is not None:
        origin = avoid_labels(origin, (tw, th), (w, h), placed, top=HUD_HEIGHT)
        placed.append(label_box(origin, (tw, th)))
    draw_text(frame, text, origin)


def draw_face(frame, reading, placed=None):
    """Outline the square the expression network saw (turned with the face) and label it with
    the expression, unless no expression is likely enough. Shares ``placed`` with draw_hand."""
    h, w = frame.shape[:2]
    corners = [(int(round(x)), int(round(y))) for x, y in reading.box.corners()]
    cv2.polylines(frame, [np.array(corners, np.int32)], True, CYAN, 2, cv2.LINE_AA)
    if reading.covered:
        text = "Face covered"
    elif reading.expression:
        text = f"{reading.expression} {reading.confidence:.0%}"
    else:
        return
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    origin = label_origin(corners, (tw, th), (w, h), top=HUD_HEIGHT)
    if placed is not None:
        origin = avoid_labels(origin, (tw, th), (w, h), placed, top=HUD_HEIGHT)
        placed.append(label_box(origin, (tw, th)))
    draw_text(frame, text, origin, CYAN)


def label_origin(pts, text_size, frame_size, margin=10, gap=15, top=10):
    """Bottom-left text origin: above the hand, or below it when there is no room, and
    always fully on screen (the spec drew it at wrist_y + 30, usually below the frame).
    ``top`` is the height of the strip at the top that labels must stay out of."""
    (tw, th), (w, h) = text_size, frame_size
    xs, ys = zip(*pts)
    x = min(max(min(xs), margin), max(margin, w - tw - margin))
    y = min(ys) - gap
    if y - th < top:
        y = max(ys) + th + gap
    y = min(max(y, th + top), h - margin)
    return int(x), int(y)


def label_box(origin, text_size, pad=6):
    """Screen rectangle (x0, y0, x1, y1) covered by a label drawn at ``origin``."""
    (x, y), (tw, th) = origin, text_size
    return x - pad, y - th - pad, x + tw + pad, y + pad


def avoid_labels(origin, text_size, frame_size, placed, margin=10, top=10):
    """Slide a label up, or else down, in whole label heights until it clears the
    labels already ``placed``, staying on screen. Returns ``origin`` if nothing fits."""
    def clashes(o):
        x0, y0, x1, y1 = label_box(o, text_size)
        return any(x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1 for bx0, by0, bx1, by1 in placed)

    th, h = text_size[1], frame_size[1]
    origin = (origin[0], min(max(origin[1], th + top), h - margin))
    step = th + 14  # taller than a label box (th + 2 * pad), so neighbours never touch
    for direction in (-1, 1):
        for k in range(len(placed) + 1):
            candidate = (origin[0], origin[1] + direction * k * step)
            if th + top <= candidate[1] <= h - margin and not clashes(candidate):
                return candidate
    return origin


def draw_hud(frame, fps, two_hand_gesture=None, mode=None, faces=None):
    """``mode``: None when there is no learned model, else whether it is in use.
    ``faces``: None when expressions are not loaded, else whether they are on."""
    status = f"{fps:4.1f} FPS   q / Esc: quit"
    if mode is not None:
        status += f"   m: model {'ON (rules + model)' if mode else 'OFF (rules only)'}"
    if faces is not None:
        status += f"   f: faces {'ON' if faces else 'OFF'}"
    draw_text(frame, status, (10, 30), WHITE, 0.6)
    if two_hand_gesture:
        text = f"Both hands: {two_hand_gesture}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        scale = min(1.0, (frame.shape[1] - 20) / tw)  # shrink to fit narrow frames
        tw = int(tw * scale)
        draw_text(frame, text, ((frame.shape[1] - tw) // 2, 70), MAGENTA, scale)


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        hint = ""
        if sys.platform == "darwin":
            hint = ("\nOn macOS, allow camera access for your terminal app in System Settings > "
                    "Privacy & Security > Camera, then restart the terminal.")
        raise RuntimeError(f"Could not open camera {index}. Try another --camera index.{hint}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def window_visible():
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:  # macOS raises instead of returning 0 once the window is gone
        return False


def run(cap, pipeline, max_frames=None, faces=None):
    """``faces``: an optional FacePipeline; its ``enabled`` flag is what the f key switches."""
    frames, read_failures, fps, window_seen = 0, 0, 0.0, False
    start = last = time.monotonic()
    while max_frames is None or frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            read_failures += 1
            if read_failures > MAX_CONSECUTIVE_READ_FAILURES:
                raise RuntimeError("The camera stopped delivering frames.")
            time.sleep(0.03)
            continue
        read_failures = 0

        frame = cv2.flip(frame, 1)  # mirror view; GesturePipeline(mirrored=True) corrects the labels
        now = time.monotonic()
        # Faces first, so phantom hands on them can be dropped; then the expressions, which
        # are not read while a hand covers the eyes or mouth.
        reading_faces = faces is not None and faces.enabled
        result = pipeline.process(frame, now, faces.detect(frame, now) if reading_faces else ())
        face_readings = faces.read(now, [h.points_px for h in result.hands]) if reading_faces else []
        placed = []
        for reading in result.hands:
            draw_hand(frame, reading, placed)
        for reading in face_readings:
            draw_face(frame, reading, placed)
        if frames:
            instant = 1.0 / max(now - last, 1e-6)
            fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
        last = now
        draw_hud(frame, fps, result.two_hand_gesture, pipeline.use_model if pipeline.model else None,
                 faces.enabled if faces is not None else None)

        cv2.imshow(WINDOW_NAME, frame)
        frames += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m") and pipeline.model is not None:
            pipeline.use_model = not pipeline.use_model
        if key == ord("f") and faces is not None:
            faces.enabled = not faces.enabled
        # Closing the window does not send a key, so check for it explicitly. Only trust a
        # "not visible" answer after the window has been seen, in case a backend lags.
        visible = window_visible()
        if window_seen and not visible:
            break
        window_seen = window_seen or visible
    elapsed = time.monotonic() - start
    return frames, (frames / elapsed if elapsed else 0.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="requested capture width")
    parser.add_argument("--height", type=int, default=720, help="requested capture height")
    parser.add_argument("--model", type=Path, help="path to a hand_landmarker.task model (default: auto-download)")
    parser.add_argument("--gesture-model", type=Path, default=DEFAULT_CLASSIFIER_PATH,
                        help="learned gesture classifier (default: models/gesture_classifier.json)")
    parser.add_argument("--rules-only", action="store_true", help="start with the learned model off (m turns it on)")
    parser.add_argument("--min-detection", type=float, default=MIN_HAND_DETECTION_CONFIDENCE,
                        help=f"MediaPipe hand detection confidence (default {MIN_HAND_DETECTION_CONFIDENCE}; raise if phantom hands appear)")
    parser.add_argument("--min-presence", type=float, default=MIN_HAND_PRESENCE_CONFIDENCE,
                        help=f"MediaPipe hand presence confidence (default {MIN_HAND_PRESENCE_CONFIDENCE})")
    parser.add_argument("--no-faces", action="store_true", help="no facial expressions (the face models are not loaded)")
    parser.add_argument("--max-faces", type=int, default=1, help="faces to read expressions from (default 1)")
    parser.add_argument("--min-expression-confidence", type=float, default=MIN_EXPRESSION_CONFIDENCE,
                        help=f"show an expression only when at least this likely (default {MIN_EXPRESSION_CONFIDENCE})")
    parser.add_argument("--max-frames", type=int, help="stop after N frames (for testing)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.model and not args.model.is_file():
        raise RuntimeError(f"Model file not found: {args.model}")
    model_path = args.model or ensure_model()
    gesture_model = None
    if args.gesture_model.is_file():
        gesture_model = GestureModel.load(args.gesture_model)
    else:
        print(f"No learned gesture model at {args.gesture_model}; using the rules alone.")
    face_paths = None if args.no_faces else ensure_face_models()
    cap = open_camera(args.camera, args.width, args.height)
    try:
        with ExitStack() as stack:
            landmarker = stack.enter_context(
                create_landmarker(model_path, min_detection=args.min_detection, min_presence=args.min_presence))
            faces = None
            if face_paths:
                face_landmarker = stack.enter_context(create_face_landmarker(face_paths[0], num_faces=args.max_faces))
                faces = FacePipeline(face_landmarker, ExpressionModel(face_paths[1]),
                                     min_confidence=args.min_expression_confidence)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Tracking active on camera {args.camera} at {w}x{h}. Press 'q' or Esc to exit.")
            pipeline = GesturePipeline(landmarker, mirrored=True, model=gesture_model,
                                       confirm_frames=HAND_CONFIRM_FRAMES)
            pipeline.use_model = gesture_model is not None and not args.rules_only
            frames, fps = run(cap, pipeline, args.max_frames, faces)
            print(f"Processed {frames} frames at {fps:.1f} FPS.")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
