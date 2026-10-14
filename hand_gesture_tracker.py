"""Real-time webcam hand tracking and gesture recognition (MediaPipe Tasks + OpenCV).

Usage:  python hand_gesture_tracker.py [--camera 0] [--width 1280] [--height 720]
Quit:   press q or Esc, or close the window.
"""

import argparse
import hashlib
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi
import cv2
import mediapipe as mp

from gesture_logic import (
    WRIST,
    HandTracker,
    classify_gesture,
    finger_states,
    pinch_ratio,
    thumb_direction,
)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

WINDOW_NAME = "MediaPipe Hand Gesture Tracker"
HAND_CONNECTIONS = [(c.start, c.end) for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS]
FINGERTIPS = {4, 8, 12, 16, 20}
HUD_HEIGHT = 45  # top strip kept free of hand labels
MAX_CONSECUTIVE_READ_FAILURES = 60  # ~2 s at 30 fps; cameras often drop frames while warming up

GREEN, WHITE, BLACK, ORANGE = (0, 255, 0), (255, 255, 255), (0, 0, 0), (0, 165, 255)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path=DEFAULT_MODEL_PATH):
    """Download the pinned hand-landmarker model on first use and verify its checksum."""
    path = Path(path)
    if path.exists() and _sha256(path) == MODEL_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"Downloading hand landmark model to {path} ...")
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(MODEL_URL, context=context, timeout=60) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the model ({e}). Download it manually from\n  {MODEL_URL}\nto\n  {path}") from e
    if _sha256(tmp) != MODEL_SHA256:
        tmp.unlink()
        raise RuntimeError("Downloaded model failed its SHA-256 check; refusing to use it.")
    tmp.replace(path)
    return path


def create_landmarker(model_path, num_hands=2):
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
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
    gesture: str  # majority-voted label shown on screen
    raw_gesture: str  # this frame's label
    fingers: list  # [thumb, index, middle, ring, pinky]
    pinch: float  # thumb-to-index tip distance in palm lengths
    points_px: list  # 21 smoothed (x, y) image points in pixels


class GesturePipeline:
    """Runs the landmarker on one frame and turns its output into per-hand readings."""

    def __init__(self, landmarker, mirrored=True, tracker=None):
        self.landmarker = landmarker
        self.mirrored = mirrored
        self.tracker = tracker or HandTracker()
        self._last_timestamp_ms = -1

    def process(self, frame_bgr, t_seconds):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # VIDEO mode rejects timestamps that do not strictly increase.
        timestamp_ms = max(int(t_seconds * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

        image_hands = [[(p.x, p.y, p.z) for p in hand] for hand in result.hand_landmarks]
        world_hands = [[(p.x, p.y, p.z) for p in hand] for hand in result.hand_world_landmarks]
        labels = [categories[0].category_name for categories in result.handedness]
        tracks = self.tracker.update([hand[WRIST][:2] for hand in image_hands])

        readings = []
        for track, image_pts, world_pts, label in zip(tracks, image_hands, world_hands, labels, strict=True):
            points_px = [(x * w, y * h) for x, y, _ in track.smoother(image_pts, t_seconds)]
            fingers = finger_states(world_pts)
            pinch = pinch_ratio(world_pts)
            raw = classify_gesture(fingers, pinch, thumb_direction(points_px))
            readings.append(HandReading(
                track_id=track.id,
                handedness=user_handedness(label, self.mirrored),
                gesture=track.vote.update(raw),
                raw_gesture=raw,
                fingers=fingers,
                pinch=pinch,
                points_px=points_px,
            ))
        return readings


def draw_text(frame, text, origin, color=GREEN, scale=0.8):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 5, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_hand(frame, reading):
    h, w = frame.shape[:2]
    pts = [(int(round(x)), int(round(y))) for x, y in reading.points_px]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], WHITE, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 6 if i in FINGERTIPS else 4, ORANGE if i in FINGERTIPS else GREEN, -1, cv2.LINE_AA)

    text = f"{reading.handedness}: {reading.gesture}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    draw_text(frame, text, label_origin(pts, (tw, th), (w, h), top=HUD_HEIGHT))


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


def draw_hud(frame, fps):
    draw_text(frame, f"{fps:4.1f} FPS   q / Esc: quit", (10, 30), WHITE, 0.6)


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


def run(cap, pipeline, max_frames=None):
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
        readings = pipeline.process(frame, now)
        for reading in readings:
            draw_hand(frame, reading)
        if frames:
            instant = 1.0 / max(now - last, 1e-6)
            fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
        last = now
        draw_hud(frame, fps)

        cv2.imshow(WINDOW_NAME, frame)
        frames += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
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
    parser.add_argument("--max-frames", type=int, help="stop after N frames (for testing)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.model and not args.model.is_file():
        raise RuntimeError(f"Model file not found: {args.model}")
    model_path = args.model or ensure_model()
    cap = open_camera(args.camera, args.width, args.height)
    try:
        with create_landmarker(model_path) as landmarker:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Hand tracking active on camera {args.camera} at {w}x{h}. Press 'q' or Esc to exit.")
            frames, fps = run(cap, GesturePipeline(landmarker, mirrored=True), args.max_frames)
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
