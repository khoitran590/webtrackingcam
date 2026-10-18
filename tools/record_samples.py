"""Record your own examples of each gesture and expression, to recalibrate on your camera.

The window shows the camera like the app does. Press a class key: after a short countdown
it records for a few seconds, one file per take. Gesture keys record the landmarks of every
visible hand (samples/), for tools/train_gesture_model.py; expression keys record what the
expression network makes of your face (samples/faces/), for tools/fit_expression_offsets.py.
Both then learn from your takes as well as the public photos and report on them separately.

- Show only the hand making the gesture; rest the other one out of view.
- Move the hand slowly during a take (tilt, turn, nearer, further) so the frames differ.
- For "no gesture", just use your hands naturally: resting, typing, reaching, touching your face.
- For expressions, hold each one naturally, keeping your hands off your face. "Neutral" is
  your resting face (looking at the screen as you would while using the app) and matters most.
- 3-5 takes per class, on different days or in different light, is a good start.

    .venv/bin/python tools/record_samples.py [--camera 0] [--save-extra-hands]
    .venv/bin/python tools/train_gesture_model.py        # retrain with your gesture takes included
    .venv/bin/python tools/fit_expression_offsets.py     # refit the expression offsets with your face takes

Keys: the class keys shown on screen; q or Esc quits.
"""

import argparse
import json
import sys
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from face_expression import CLASS_OFFSETS, ExpressionModel, FacePipeline, create_face_landmarker  # noqa: E402
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
    ensure_face_models,
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
# key -> (face_expression.EXPRESSIONS entry, what to show)
FACE_CLASSES = {
    "n": ("neutral", "Neutral (resting face)"), "h": ("happy", "Happy"), "s": ("sad", "Sad"),
    "w": ("surprise", "Surprised"), "a": ("anger", "Angry"), "d": ("disgust", "Disgusted"), "f": ("fear", "Fearful"),
}
FACE_SET = {cls for cls, _ in FACE_CLASSES.values()}


class Recorder:
    """Countdown, then collect every visible hand for ``duration`` seconds, then save the take."""

    def __init__(self, out_dir, countdown=2.0, duration=4.0, face_offsets=CLASS_OFFSETS, extra_hands_dir=None):
        """``extra_hands_dir``: if given, frames of a gesture take that show more than one hand are
        saved there (camera image + every hand's landmarks and scores), to study phantom hands."""
        self.out_dir, self.countdown, self.duration = Path(out_dir), countdown, duration
        self.face_offsets = np.asarray(face_offsets, dtype=float)
        self.extra_hands_dir = Path(extra_hands_dir) if extra_hands_dir else None
        self.cls = self.started = self.take_name = None
        self.rows = []
        self.saved = []  # paths of finished takes
        self.reports = []  # one line per finished take, also printed as it happens
        self._reset()

    def _reset(self):
        self.frames = self.found = self.covered = self.extra_hands = 0  # frames seen while recording
        self.extra_saved = 0  # frames written to extra_hands_dir
        self.shown = Counter()  # what the app showed for the face or hand being recorded

    def start(self, cls, now):
        if self.cls is None:
            self.cls, self.started, self.rows = cls, now, []
            self.take_name = f"{cls}-{time.strftime('%Y%m%d-%H%M%S')}"
            self._reset()
            print(f"Take started: {self.cls}", flush=True)

    def update(self, reading, frame_size, now, faces=(), frame=None):
        """Feed one frame's FrameReading and FaceReadings, and the camera frame before anything is
        drawn on it (needed only with ``extra_hands_dir``). Returns the status line to show, or None when idle."""
        if self.cls is None:
            return None
        elapsed = now - self.started
        if elapsed < self.countdown:
            return f"Get ready: {self.cls} in {self.countdown - elapsed:.0f}"
        if elapsed < self.countdown + self.duration:
            self.frames += 1
            if self.cls in FACE_SET:
                self.found += bool(faces)
                for face in faces:
                    self.shown[face.expression or ("covered" if face.covered else "no label")] += 1
                    self.covered += bool(face.covered)
                    if face.raw_probabilities is not None:  # None while a hand covers the face
                        # The network's own log-probabilities (up to a constant), without the
                        # offsets the app adds, so offsets can be refitted from them.
                        scores = np.log(np.clip(face.raw_probabilities, 1e-12, 1.0)) - self.face_offsets
                        self.rows.append({"class": self.cls, "scores": [round(float(v), 5) for v in scores]})
                return f"RECORDING {self.cls}: {len(self.rows)} faces"
            self.found += bool(reading.hands)
            self.extra_hands += len(reading.hands) > 1
            for hand in reading.hands:
                self.shown[hand.gesture or "no label"] += 1
                self.rows.append({"class": self.cls, "image_label": hand.image_label, "size": list(frame_size),
                                  "points_px": [list(p) for p in hand.points_px], "world": [list(p) for p in hand.world_points],
                                  "track_id": hand.track_id})
            if self.extra_hands_dir and frame is not None and len(reading.hands) > 1 and self.cls != "none":
                self._save_extra_hands(reading, frame)
            return f"RECORDING {self.cls}: {len(self.rows)} hands"
        self._save()
        return None

    def report(self):
        """One line on the take just finished: what was kept, how the app read it now, and any problem."""
        kind = "faces" if self.cls in FACE_SET else "hands"
        frames = max(self.frames, 1)
        total = sum(self.shown.values()) or 1
        reads = ", ".join(f"{label} {n / total:.0%}" for label, n in self.shown.most_common(3))
        line = f"TAKE {self.cls}: {len(self.rows)} {kind} kept from {self.frames} frames; app showed: {reads or 'nothing'}"
        problems = []
        if self.found / frames < 0.8:
            problems.append(f"{'face' if kind == 'faces' else 'hand'} found in only {self.found / frames:.0%} of frames")
        if self.covered / frames > 0.2:
            problems.append(f"a hand covered the face in {self.covered / frames:.0%} of frames")
        if self.extra_hands / frames > 0.2 and self.cls != "none":  # both hands are fine for "no gesture"
            problems.append(f"two hands in view in {self.extra_hands / frames:.0%} of frames; rest the other one out of view")
        if not self.rows:
            problems.append("nothing saved")
        if self.extra_saved:
            line += f"\n  {self.extra_saved} frames with extra hands saved to {self.extra_hands_dir / self.take_name}"
        return line + "".join(f"\n  WARNING: {p}" for p in problems)

    def _save_extra_hands(self, reading, frame):
        folder = self.extra_hands_dir / self.take_name
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"{self.frames:03d}"
        cv2.imwrite(str(stem.with_suffix(".jpg")), frame)
        hands = [{"track_id": h.track_id, "image_label": h.image_label, "label_score": h.label_score,
                  "gesture": h.gesture, "raw_gesture": h.raw_gesture, "model_guess": h.model_guess,
                  "points_px": [list(p) for p in h.points_px], "world": [list(p) for p in h.world_points]}
                 for h in reading.hands]
        stem.with_suffix(".json").write_text(json.dumps({"class": self.cls, "hands": hands}))
        self.extra_saved += 1

    def _save(self):
        line = self.report()
        self.reports.append(line)
        print(line, flush=True)
        if self.rows:
            out_dir = face_dir(self.out_dir) if self.cls in FACE_SET else self.out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{self.cls}-{time.strftime('%Y%m%d-%H%M%S')}"
            path, n = out_dir / f"{stem}.jsonl", 2
            while path.exists():
                path, n = out_dir / f"{stem}-{n}.jsonl", n + 1
            path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
            self.saved.append(path)
            print(f"  saved to {path}", flush=True)
        self.cls = None


def face_dir(out_dir):
    return Path(out_dir) / "faces"


def load_takes(out_dir):
    """{take file: rows} for every recorded take."""
    return {path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for path in sorted(Path(out_dir).glob("*.jsonl"))}


def takes_per_class(out_dir):
    counts = {}
    for rows in [*load_takes(out_dir).values(), *load_takes(face_dir(out_dir)).values()]:
        if rows:
            counts[rows[0]["class"]] = counts.get(rows[0]["class"], 0) + 1
    return counts


def draw_menu(frame, counts, status, faces=True):
    h, w = frame.shape[:2]
    columns = [(10, [f"{key}: {name} ({counts.get(cls, 0)} takes)" for key, (cls, name) in CLASSES.items()]
                + ["q / Esc: quit"])]
    if faces:
        columns.append((w - 330, ["Expressions:"] + [f"{key}: {name} ({counts.get(cls, 0)} takes)"
                                                     for key, (cls, name) in FACE_CLASSES.items()]))
    for x, lines in columns:
        for i, line in enumerate(lines):
            draw_text(frame, line, (x, h - 20 - 24 * (len(lines) - 1 - i)), WHITE, 0.55)
    if status:
        draw_text(frame, status, (10, 70), MAGENTA, 1.0)


def record(cap, pipeline, recorder, max_frames=None, clock=time.monotonic, faces=None):
    """``faces``: a FacePipeline, needed for the expression keys."""
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
        reading = pipeline.process(frame, now, faces.detect(frame, now) if faces else ())
        face_readings = faces.read(now, [h.points_px for h in reading.hands]) if faces else []
        saved = len(recorder.saved)
        status = recorder.update(reading, (frame.shape[1], frame.shape[0]), now, face_readings, frame)
        if len(recorder.saved) != saved:
            counts = takes_per_class(recorder.out_dir)
        placed = []
        for hand in reading.hands:
            draw_hand(frame, hand, placed)
        draw_menu(frame, counts, status, faces is not None)
        cv2.imshow(WINDOW_NAME, frame)
        frames += 1
        if frames == 1:
            print("Recorder ready: click the camera window, then press a class key.", flush=True)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print(f"Quit key pressed ({'q' if key == ord('q') else 'Esc'}).", flush=True)
            if recorder.cls is not None:
                print(f"  Take of {recorder.cls} discarded: it had not finished.", flush=True)
            break
        if frames > 1 and not window_visible():
            print("Window closed.", flush=True)
            break
        if key != 255 and chr(key) not in CLASSES and chr(key) not in FACE_CLASSES:
            print(f"Key {chr(key)!r} is not a class key.", flush=True)
        if chr(key) in CLASSES:
            recorder.start(CLASSES[chr(key)][0], now)
        elif chr(key) in FACE_CLASSES and faces is not None:
            recorder.start(FACE_CLASSES[chr(key)][0], now)
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", type=Path, default=SAMPLES)
    parser.add_argument("--no-faces", action="store_true", help="gestures only")
    parser.add_argument("--save-extra-hands", action="store_true",
                        help="save frames of gesture takes that show more than one hand to <out>/extra_hands")
    args = parser.parse_args(argv)

    model = GestureModel.load(DEFAULT_CLASSIFIER_PATH) if DEFAULT_CLASSIFIER_PATH.exists() else None
    face_paths = None if args.no_faces else ensure_face_models()
    cap = open_camera(args.camera, args.width, args.height)
    try:
        with ExitStack() as stack:
            landmarker = stack.enter_context(create_landmarker(ensure_model()))
            faces = None
            if face_paths:
                faces = FacePipeline(stack.enter_context(create_face_landmarker(face_paths[0])), ExpressionModel(face_paths[1]))
            recorder = Recorder(args.out, extra_hands_dir=args.out / "extra_hands" if args.save_extra_hands else None)
            record(cap, GesturePipeline(landmarker, mirrored=True, model=model, confirm_frames=HAND_CONFIRM_FRAMES),
                   recorder, faces=faces)
    except KeyboardInterrupt:
        pass
    else:
        print(f"Session over: {len(recorder.saved)} takes saved; all takes so far: "
              f"{dict(sorted(takes_per_class(args.out).items()))}", flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
