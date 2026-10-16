"""Record your own examples of each gesture for tools/train_gesture_model.py.

The window shows the camera like the app does. Press a class key: after a short countdown
it records the landmarks of every visible hand for a few seconds, one file per take in
samples/. Retraining then learns from your hands and your camera as well as from HaGRID,
and reports accuracy on your own takes separately.

- Show only the hand making the gesture; rest the other one out of view.
- Move the hand slowly during a take (tilt, turn, nearer, further) so the frames differ.
- For "no gesture", just use your hands naturally: resting, typing, reaching, touching your face.
- 3-5 takes per class, on different days or in different light, is a good start.

    .venv/bin/python tools/record_samples.py [--camera 0]
    .venv/bin/python tools/train_gesture_model.py        # retrain with your samples included

Keys: the class keys shown on screen; q or Esc quits.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gesture_model import DEFAULT_CLASSIFIER_PATH, GestureModel  # noqa: E402
from hand_gesture_tracker import (  # noqa: E402
    HAND_CONFIRM_FRAMES,
    MAGENTA,
    MAX_CONSECUTIVE_READ_FAILURES,
    WHITE,
    WINDOW_NAME,
    GesturePipeline,
    create_landmarker,
    draw_hand,
    draw_text,
    ensure_model,
    open_camera,
    window_visible,
)

SAMPLES = ROOT / "samples"
# key -> (model class, what to show)
CLASSES = {
    "1": ("ok", "OK Sign"), "2": ("rock", "Rock On / Spider-Man ILY"), "3": ("call", "Call Me"),
    "4": ("l_shape", "L Shape"), "5": ("thumbs", "Thumbs up / down / out"), "6": ("fist", "Fist"),
    "7": ("palm", "Open Palm"), "8": ("peace", "Peace / Victory"), "9": ("one", "Pointing"),
    "0": ("none", "No gesture (relaxed, natural hands)"), "o": ("other", "Any other hand shape"),
}


class Recorder:
    """Countdown, then collect every visible hand for ``duration`` seconds, then save the take."""

    def __init__(self, out_dir, countdown=2.0, duration=4.0):
        self.out_dir, self.countdown, self.duration = Path(out_dir), countdown, duration
        self.cls = self.started = None
        self.rows = []
        self.saved = []  # paths of finished takes

    def start(self, cls, now):
        if self.cls is None:
            self.cls, self.started, self.rows = cls, now, []

    def update(self, reading, frame_size, now):
        """Feed one frame's FrameReading. Returns the status line to show, or None when idle."""
        if self.cls is None:
            return None
        elapsed = now - self.started
        if elapsed < self.countdown:
            return f"Get ready: {self.cls} in {self.countdown - elapsed:.0f}"
        if elapsed < self.countdown + self.duration:
            for hand in reading.hands:
                self.rows.append({"class": self.cls, "image_label": hand.image_label, "size": list(frame_size),
                                  "points_px": [list(p) for p in hand.points_px], "world": [list(p) for p in hand.world_points]})
            return f"RECORDING {self.cls}: {len(self.rows)} hands"
        self._save()
        return None

    def _save(self):
        if self.rows:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{self.cls}-{time.strftime('%Y%m%d-%H%M%S')}"
            path, n = self.out_dir / f"{stem}.jsonl", 2
            while path.exists():
                path, n = self.out_dir / f"{stem}-{n}.jsonl", n + 1
            path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
            self.saved.append(path)
            print(f"Saved {len(self.rows)} hands of '{self.cls}' to {path}")
        self.cls = None


def load_takes(out_dir):
    """{take file: rows} for every recorded take."""
    return {path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for path in sorted(Path(out_dir).glob("*.jsonl"))}


def takes_per_class(out_dir):
    counts = {}
    for rows in load_takes(out_dir).values():
        if rows:
            counts[rows[0]["class"]] = counts.get(rows[0]["class"], 0) + 1
    return counts


def draw_menu(frame, counts, status):
    h = frame.shape[0]
    lines = [f"{key}: {name} ({counts.get(cls, 0)} takes)" for key, (cls, name) in CLASSES.items()]
    for i, line in enumerate(lines + ["q / Esc: quit"]):
        draw_text(frame, line, (10, h - 20 - 24 * (len(lines) - i)), WHITE, 0.55)
    if status:
        draw_text(frame, status, (10, 70), MAGENTA, 1.0)


def record(cap, pipeline, recorder, max_frames=None, clock=time.monotonic):
    frames, failures, counts = 0, 0, takes_per_class(recorder.out_dir)
    while max_frames is None or frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > MAX_CONSECUTIVE_READ_FAILURES:
                raise RuntimeError("The camera stopped delivering frames.")
            time.sleep(0.03)
            continue
        failures = 0
        frame = cv2.flip(frame, 1)  # mirrored like the app, so samples match what the app sees
        now = clock()
        reading = pipeline.process(frame, now)
        saved = len(recorder.saved)
        status = recorder.update(reading, (frame.shape[1], frame.shape[0]), now)
        if len(recorder.saved) != saved:
            counts = takes_per_class(recorder.out_dir)
        placed = []
        for hand in reading.hands:
            draw_hand(frame, hand, placed)
        draw_menu(frame, counts, status)
        cv2.imshow(WINDOW_NAME, frame)
        frames += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or (frames > 1 and not window_visible()):
            break
        if chr(key) in CLASSES:
            recorder.start(CLASSES[chr(key)][0], now)
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", type=Path, default=SAMPLES)
    args = parser.parse_args(argv)

    model = GestureModel.load(DEFAULT_CLASSIFIER_PATH) if DEFAULT_CLASSIFIER_PATH.exists() else None
    cap = open_camera(args.camera, args.width, args.height)
    try:
        with create_landmarker(ensure_model()) as landmarker:
            record(cap, GesturePipeline(landmarker, mirrored=True, model=model, confirm_frames=HAND_CONFIRM_FRAMES), Recorder(args.out))
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
