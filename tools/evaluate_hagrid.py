"""Measure gesture accuracy on real photos from HaGRID v2 (test split).

Downloads 50-100 photos per class (one parquet row group, ~11 MB) with HTTP range requests,
runs each through the app's own GesturePipeline, and prints per-class accuracy plus
false alarms, for the rules alone and for rules + learned model side by side. The
thresholds in gesture_logic.py were tuned with --shard-offset 0 and checked with
--shard-offset 3 (different shards, held out). The learned model is trained on the
train split (different people), so both offsets are held out for it.

    uv pip install -r requirements-eval.txt
    .venv/bin/python tools/evaluate_hagrid.py                   # calibration photos
    .venv/bin/python tools/evaluate_hagrid.py --shard-offset 3  # held-out photos

HaGRID v2 is published under its own license (a variant of CC BY-SA 4.0):
https://github.com/hukenovs/hagrid. Photos are cached locally and not redistributed.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gesture_model import DEFAULT_CLASSIFIER_PATH, GestureModel  # noqa: E402
from hagrid_data import fetch  # noqa: E402
from hand_gesture_tracker import (  # noqa: E402
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    GesturePipeline,
    ensure_model,
)

# HaGRID class -> label(s) any detected hand should show
SINGLE = {"ok": {"OK Sign"}, "rock": {"Rock On", "Spider-Man / ILY"}, "call": {"Call Me"}, "thumb_index": {"L Shape"},
          "like": {"Thumbs Up"}, "dislike": {"Thumbs Down"}, "fist": {"Fist"}, "palm": {"Open Palm"},
          "peace": {"Peace / Victory"}, "one": {"Pointing"}}
# Variants HaGRID names separately but the app shows under the same label
VARIANTS = {"stop": {"Open Palm"}, "stop_inverted": {"Open Palm"}, "two_up": {"Peace / Victory"},
            "two_up_inverted": {"Peace / Victory"}, "peace_inverted": {"Peace / Victory"}, "point": {"Pointing"},
            "mute": {"Pointing"}}
# Hand shapes the app has no name for: a finger count ("3 Fingers Up") or no label is right
OTHER = ["four", "three", "three2", "three3", "three_gun", "little_finger", "middle_finger", "grabbing", "grip"]
# HaGRID class -> expected two-hand gesture
PAIR = {"hand_heart": "Heart", "hand_heart2": "Heart", "take_picture": "Frame", "timeout": "Time Out",
        "xsign": "X Sign", "holy": "Prayer", "thumb_index2": "Double L Shape"}
MODES = ("rules", "rules + model")


class StillImages:
    """Lets GesturePipeline (written for video) run on unrelated photos: no tracking between them."""

    def __init__(self, landmarker):
        self.landmarker = landmarker

    def detect_for_video(self, image, timestamp_ms):
        return self.landmarker.detect(image)


def named(label):
    """A gesture name, as opposed to no label or a finger count."""
    return label is not None and not label[0].isdigit()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard-offset", type=int, default=0, help="0 = calibration photos, 3 = held-out photos")
    parser.add_argument("--cache", type=Path, default=ROOT / "tools" / ".cache" / "hagrid")
    parser.add_argument("--classes", default=",".join([*SINGLE, *VARIANTS, "no_gesture", *OTHER, *PAIR]))
    parser.add_argument("--gesture-model", type=Path, default=DEFAULT_CLASSIFIER_PATH)
    parser.add_argument("--min-detection", type=float, default=MIN_HAND_DETECTION_CONFIDENCE,
                        help="MediaPipe min_hand_detection_confidence (default: the app's)")
    parser.add_argument("--min-presence", type=float, default=MIN_HAND_PRESENCE_CONFIDENCE,
                        help="MediaPipe min_hand_presence_confidence (default: the app's)")
    parser.add_argument("--save", type=Path, help="also write every photo's labels to this JSON file")
    args = parser.parse_args()

    model = GestureModel.load(args.gesture_model) if args.gesture_model.is_file() else None
    modes = MODES if model else MODES[:1]
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=args.min_detection, min_hand_presence_confidence=args.min_presence)

    # mode -> class -> one row per photo: (each hand's label, both hands detected, two-hand label)
    results = {mode: defaultdict(list) for mode in modes}
    saved = {}
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for cls in args.classes.split(","):
            folder = args.cache / f"offset{args.shard_offset}" / cls
            fetch(cls, folder, args.shard_offset)
            for path in sorted(folder.glob("*.jpg")):
                frame = cv2.imread(str(path))
                for mode in modes:
                    pipeline = GesturePipeline(StillImages(landmarker), mirrored=False, model=model)
                    pipeline.use_model = mode != "rules"
                    result = pipeline.process(frame, 0.0)
                    row = ([h.raw_gesture for h in result.hands], len(result.hands) == 2, result.two_hand_gesture)
                    results[mode][cls].append(row)
                    saved.setdefault(f"{cls}/{path.name}", {})[mode] = row
            print(f"  {cls}: {len(results[modes[0]][cls])} photos", file=sys.stderr)
    if args.save:
        args.save.write_text(json.dumps(saved, indent=1))

    def table(title, classes, score, describe=lambda cls: ""):
        shown = [cls for cls in classes if cls in results[modes[0]]]
        if not shown:
            return
        print(f"{title}\n  {'':16s}{''.join(f'{m:>16s}' for m in modes)}")
        for cls in shown:
            cells = "".join(f"{score(cls, results[m][cls]):16.0%}" for m in modes)
            print(f"  {cls:16s}{cells}   {describe(cls)}")

    def hit(expected):
        return lambda cls, rows: sum(bool(expected[cls] & set(labels)) for labels, _, _ in rows) / len(rows)

    def top(cls):
        counts = Counter(label for labels, _, _ in results[modes[-1]][cls] for label in labels)
        return "most common: " + ", ".join(f"{label} ({n})" for label, n in counts.most_common(3))

    table("Single-hand gestures (hit = any detected hand shows the gesture)", SINGLE, hit(SINGLE),
          lambda cls: " / ".join(sorted(SINGLE[cls])))
    table("Variants shown under the same label", VARIANTS, hit(VARIANTS), lambda cls: " / ".join(sorted(VARIANTS[cls])))

    def false_name(cls, rows):  # photos where some hand is given a gesture name
        return sum(any(named(label) for label in labels) for labels, _, _ in rows) / len(rows)

    table("False alarms: photos where a hand is given a gesture name (lower is better)", ["no_gesture", *OTHER],
          false_name, top)
    print("Two-hand gestures (of all photos; both hands were detected in the % shown last)")
    for cls, expected in PAIR.items():
        if cls in results[modes[0]]:
            rows = {m: results[m][cls] for m in modes}
            cells = "".join(f"{sum(r[2] == expected for r in rows[m]) / len(rows[m]):16.0%}" for m in modes)
            both = sum(r[1] for r in rows[modes[0]]) / len(rows[modes[0]])
            print(f"  {cls:16s}{cells}   {expected} (both hands found in {both:.0%})")
    for mode in modes:
        false_positives = Counter((two, cls) for cls, rows in results[mode].items() for _, _, two in rows
                                  if two and not two.startswith("Double") and PAIR.get(cls) != two)
        print(f"Two-hand false positives on other classes ({mode}): {dict(false_positives) or 'none'}"
              f" (in {sum(len(r) for r in results[mode].values())} photos)")


if __name__ == "__main__":
    main()
