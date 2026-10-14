"""Train the hand-shape classifier that works alongside the rules in gesture_logic.py.

Photos come from the HaGRID v2 *train* split, which shows different people from the test
photos that tools/evaluate_hagrid.py scores, so those scores stay honest. Photos are
decoded in memory and only MediaPipe's landmarks are kept (tools/.cache/train_landmarks).
Takes recorded with tools/record_samples.py (samples/) are added, and reported separately.

    uv pip install -r requirements-train.txt
    .venv/bin/python tools/train_gesture_model.py          # writes models/gesture_classifier.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "tools" / ".cache" / "train_landmarks"

# HaGRID class -> number of 100-photo row groups to read from the train split. The
# gestures the app names get the most; "no_gesture" more again, because idle hands vary
# most; other hand shapes teach the model what *not* to call a named gesture.
ROW_GROUPS = {
    **dict.fromkeys(["ok", "rock", "call", "thumb_index", "like", "dislike", "fist", "palm", "peace", "one"], 5),
    "no_gesture": 8,
    **dict.fromkeys(["stop", "stop_inverted", "peace_inverted", "two_up", "two_up_inverted", "point", "mute",
                     "four", "three", "three2", "three3", "three_gun", "little_finger", "middle_finger",
                     "grabbing", "grip"], 3),
}

# HaGRID class -> model class. Shapes the app shows under one label share a class (stop
# and two_up are flat-hand and two-finger variants); shapes the app has no name for are
# "other", so the model learns not to call them a named gesture.
MODEL_CLASS = {
    "ok": "ok", "rock": "rock", "call": "call", "thumb_index": "l_shape", "like": "thumbs", "dislike": "thumbs",
    "fist": "fist", "palm": "palm", "stop": "palm", "stop_inverted": "palm", "peace": "peace",
    "peace_inverted": "peace", "two_up": "peace", "two_up_inverted": "peace", "one": "one", "point": "one",
    "mute": "one", "no_gesture": "none",
    **dict.fromkeys(["four", "three", "three2", "three3", "three_gun", "little_finger", "middle_finger",
                     "grabbing", "grip"], "other"),
}
# HaGRID class -> the label(s) a correct answer shows. None = no label (not a gesture);
# COUNT = any unnamed label ("3 Fingers Up") or no label.
COUNT = "count"
EXPECTED = {
    "ok": {"OK Sign"}, "rock": {"Rock On", "Spider-Man / ILY"}, "call": {"Call Me"}, "thumb_index": {"L Shape"},
    "like": {"Thumbs Up"}, "dislike": {"Thumbs Down"}, "fist": {"Fist"}, "palm": {"Open Palm"},
    "stop": {"Open Palm"}, "stop_inverted": {"Open Palm"}, "peace": {"Peace / Victory"},
    "peace_inverted": {"Peace / Victory"}, "two_up": {"Peace / Victory"}, "two_up_inverted": {"Peace / Victory"},
    "one": {"Pointing"}, "point": {"Pointing"}, "mute": {"Pointing"}, "no_gesture": {None},
    **{c: {COUNT} for c, m in MODEL_CLASS.items() if m == "other"},
}


# Your own takes (tools/record_samples.py) are named "mine:<model class>".
MINE_EXPECTED = {"ok": {"OK Sign"}, "rock": {"Rock On", "Spider-Man / ILY"}, "call": {"Call Me"},
                 "l_shape": {"L Shape"}, "thumbs": {"Thumbs Up", "Thumbs Down", "Thumb Out"}, "fist": {"Fist"},
                 "palm": {"Open Palm"}, "peace": {"Peace / Victory"}, "one": {"Pointing"}, "none": {None},
                 "other": {COUNT}}
MODEL_CLASS.update({f"mine:{c}": c for c in MINE_EXPECTED})
EXPECTED.update({f"mine:{c}": labels for c, labels in MINE_EXPECTED.items()})


def is_correct(label, hagrid_class):
    expected = EXPECTED[hagrid_class]
    return label in expected or (COUNT in expected and (label is None or label[0].isdigit()))


def detect_hands(landmarker, jpeg_bytes):
    """MediaPipe's landmarks for every hand in one photo, as plain lists."""
    import cv2
    import mediapipe as mp
    import numpy as np

    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return {"size": [bgr.shape[1], bgr.shape[0]], "hands": [
        {"label": h[0].category_name, "score": h[0].score,
         "image": [[p.x, p.y, p.z] for p in il], "world": [[p.x, p.y, p.z] for p in wl]}
        for il, wl, h in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness, strict=True)]}


def extract_landmarks(classes=ROW_GROUPS, cache=CACHE):
    """Download each class's row groups and cache their landmarks, one JSON file per group."""
    from concurrent.futures import ThreadPoolExecutor

    import hagrid_data
    import mediapipe as mp
    from hagrid_data import read_row_group, train_row_groups

    from hand_gesture_tracker import ensure_model

    todo = {cls: count for cls, count in classes.items() if not (cache / cls / ".complete").exists()}
    if not todo:
        return
    cache.mkdir(parents=True, exist_ok=True)
    # Shard footers already read, so a rerun does not ask Hugging Face again (it rate-limits).
    stats_file = cache / "shard_stats.json"
    hagrid_data.load_stats(stats_file)
    bands = {}
    try:
        for cls in sorted(todo, key=hagrid_data.CLASSES.index):  # in order, so searches share footers
            bands[cls] = hagrid_data.class_shards(cls, "train")
            train_row_groups(cls, todo[cls], bands[cls])  # reads the footers of the chosen shards
    finally:
        hagrid_data.save_stats(stats_file)

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5)
    with vision.HandLandmarker.create_from_options(options) as landmarker, ThreadPoolExecutor(4) as pool:
        for cls, count in todo.items():
            folder = cache / cls
            folder.mkdir(parents=True, exist_ok=True)
            groups = train_row_groups(cls, count, bands[cls])
            downloads = pool.map(lambda sg: read_row_group("train", *sg), groups)
            for (shard, group), images in zip(groups, downloads, strict=True):
                photos = [detect_hands(landmarker, jpeg) for jpeg in images]
                (folder / f"s{shard:04d}_g{group}.json").write_text(json.dumps(photos))
            (folder / ".complete").touch()
            print(f"  {cls}: {len(groups)} row groups, {sum(len(json.loads(p.read_text())) for p in folder.glob('*.json'))}"
                  f" photos from shards {bands[cls]}", file=sys.stderr, flush=True)


def load_hands(cache=CACHE):
    """One row per usable hand: (HaGRID class, group, image points px, world points, label).

    In a gesture photo the gesture is made by the raised hand (highest wrist); the other
    hand, if any, usually hangs at the person's side and is left out. In a no_gesture
    photo every hand is an example of "not a gesture"."""
    rows = []
    for folder in sorted(p for p in cache.iterdir() if p.is_dir() and p.name in MODEL_CLASS):
        for path in sorted(folder.glob("s*_g*.json")):
            shard = f"hagrid/{int(path.stem[1:].split('_')[0])}"
            for photo in json.loads(path.read_text()):
                w, h = photo["size"]
                hands = photo["hands"]
                if hands and folder.name != "no_gesture":
                    hands = [min(hands, key=lambda hand: hand["image"][0][1])]
                for hand in hands:
                    points_px = [(x * w, y * h) for x, y, _ in hand["image"]]
                    rows.append((folder.name, shard, points_px, hand["world"], hand["label"]))
    return rows


def load_samples(folder):
    """Rows for your own takes, in the same form as load_hands. Each take is its own group,
    so cross-validation always tests on a take the model did not train on."""
    rows = []
    for path in sorted(Path(folder).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append((f"mine:{r['class']}", f"mine/{path.stem}", r["points_px"], r["world"], r["image_label"]))
    return rows


def rule_inputs(points_px, world):
    from gesture_logic import classify_gesture, finger_states, pinch_ratio, thumb_direction

    fingers = finger_states(world)
    thumb_dir = thumb_direction(points_px)
    return classify_gesture(fingers, pinch_ratio(points_px), thumb_dir), fingers, thumb_dir


def candidates():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    yield "logistic regression C=1", lambda: LogisticRegression(C=1.0, max_iter=5000)
    yield "logistic regression C=10", lambda: LogisticRegression(C=10.0, max_iter=5000)
    for hidden in ((64,), (64, 32), (128, 64)):
        for alpha in (1e-4, 1e-2):
            yield f"MLP {hidden} alpha={alpha:g}", lambda h=hidden, a=alpha: MLPClassifier(
                h, alpha=a, max_iter=3000, early_stopping=True, n_iter_no_change=30, random_state=0)


def standardise(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0  # constant features (the wrist, the turned middle knuckle)
    return mean, scale


def out_of_fold(make, X, y, groups, folds=5):
    """Class probabilities for every hand from a model that never saw that hand's shard."""
    import numpy as np
    from sklearn.model_selection import GroupKFold

    classes = sorted(set(y))
    proba = np.zeros((len(y), len(classes)))
    for train, test in GroupKFold(folds).split(X, y, groups):
        mean, scale = standardise(X[train])
        model = make().fit((X[train] - mean) / scale, y[train])
        proba[np.ix_(test, [classes.index(c) for c in model.classes_])] = model.predict_proba((X[test] - mean) / scale)
    return classes, proba


def balanced_accuracy(correct, hagrid):
    """Accuracy averaged so every app class counts the same: each named gesture, "none" and
    "other" (HaGRID classes are averaged within the app class they belong to; there are nine
    "other" ones). Your own takes count as classes of their own. Also returns per-HaGRID-class hits."""
    per_class = {}
    for ok, c in zip(correct, hagrid, strict=True):
        per_class.setdefault(c, []).append(ok)
    families = {}
    for c, hits in per_class.items():
        families.setdefault(c if c.startswith("mine:") else MODEL_CLASS[c], []).append(sum(hits) / len(hits))
    return sum(sum(v) / len(v) for v in families.values()) / len(families), per_class


def tune_thresholds(classes, proba, rules, hagrid, passes=3, floors=None):
    """Per-class confidence thresholds that maximise balanced accuracy of the combined labels.
    ``floors`` gives a lowest allowed threshold per class."""
    import numpy as np

    from gesture_model import combine

    best_class = [classes[i] for i in proba.argmax(axis=1)]
    confidence = proba.max(axis=1)

    def score(thresholds):
        labels = [combine(r, f, t, m, p, thresholds) for (r, f, t), m, p in zip(rules, best_class, confidence, strict=True)]
        return balanced_accuracy([is_correct(label, c) for label, c in zip(labels, hagrid, strict=True)], hagrid)[0]

    floors = floors or {}
    thresholds = {str(c): max(0.5, floors.get(c, 0.0)) for c in classes}
    best = score(thresholds)
    grid = [round(float(t), 2) for t in np.arange(0.0, 1.0001, 0.05)] + [1.01]  # 1.01 = never use the model's answer
    for _ in range(passes):
        for c in classes:
            for t in (t for t in grid if t >= floors.get(c, 0.0)):
                trial = {**thresholds, c: t}
                s = score(trial)
                if s > best + 1e-9:
                    best, thresholds = s, trial
    return thresholds, best


def export(model, mean, scale, classes, thresholds, info):
    """JSON the app reads with gesture_model.GestureModel (numpy only)."""
    from sklearn.linear_model import LogisticRegression

    from gesture_model import FEATURE_VERSION

    if isinstance(model, LogisticRegression):
        layers = [(model.coef_.T, model.intercept_)]
    else:
        layers = list(zip(model.coefs_, model.intercepts_, strict=True))
    assert list(model.classes_) == classes
    return {"feature_version": FEATURE_VERSION, "classes": classes, "mean": mean.tolist(), "scale": scale.tolist(),
            "layers": [{"weights": w.tolist(), "bias": b.tolist()} for w, b in layers],
            "thresholds": thresholds, "info": info}


def train(output, cache=CACHE, samples=None, min_none=0.8):
    import numpy as np
    import sklearn

    from gesture_model import GestureModel, hand_features

    rows = load_hands(cache)
    mine = load_samples(samples) if samples else []
    if mine:
        print(f"Including {len(mine)} hands from your takes in {samples}")
    rows += mine
    hagrid = [r[0] for r in rows]
    y = np.array([MODEL_CLASS[c] for c in hagrid], dtype=object)
    groups = np.array([r[1] for r in rows])
    X = np.array([hand_features(p, w, label) for _, _, p, w, label in rows])
    rules = [rule_inputs(p, w) for _, _, p, w, _ in rows]
    print(f"{len(rows)} hands from {len(set(groups))} shards: {dict(sorted(Counter(y).items()))}")

    rule_labels = [r[0] for r in rules]
    rules_score, rules_per_class = balanced_accuracy([is_correct(lab, c) for lab, c in zip(rule_labels, hagrid, strict=True)], hagrid)
    print(f"  rules alone: balanced accuracy {rules_score:.1%}")

    results = []
    for name, make in candidates():
        start = time.time()
        classes, proba = out_of_fold(make, X, y, groups)
        model_score = np.mean([np.mean([classes[i] == c for i, c in zip(proba.argmax(1)[y == k], y[y == k], strict=True)])
                               for k in classes])
        thresholds, combined = tune_thresholds(classes, proba, rules, hagrid, passes=1, floors={"none": min_none})
        results.append((model_score, name, make))
        print(f"  {name:32s} model classes {model_score:.1%}   combined with rules {combined:.1%}   ({time.time() - start:.0f}s)")
    # Choose by how well the model itself sorts hand shapes, not by the combined score: with
    # the "no gesture" floor, that score rewards confidence, and the larger nets were the most
    # confidently wrong on close-up open palms (P(none) up to 0.99, vs 0.77 for 64 units).
    # Candidates run from smallest to largest; differences under 0.2 points are noise, so
    # take the smallest model within that of the best.
    best = max(r[0] for r in results)
    _, name, make = next(r for r in results if r[0] >= best - 0.002)

    classes, proba = out_of_fold(make, X, y, groups)
    thresholds, combined = tune_thresholds(classes, proba, rules, hagrid, floors={"none": min_none})
    best_class = [classes[i] for i in proba.argmax(axis=1)]
    from gesture_model import combine
    labels = [combine(r, f, t, m, p, thresholds) for (r, f, t), m, p in zip(rules, best_class, proba.max(axis=1), strict=True)]
    _, per_class = balanced_accuracy([is_correct(lab, c) for lab, c in zip(labels, hagrid, strict=True)], hagrid)
    print(f"Chosen: {name}; thresholds {thresholds}")
    print(f"  out-of-fold balanced accuracy: rules {rules_score:.1%} -> combined {combined:.1%}")
    for c in sorted(per_class):
        print(f"    {c:16s} n={len(per_class[c]):4d}  rules {np.mean(rules_per_class[c]):5.1%} -> combined {np.mean(per_class[c]):5.1%}")
    mine_classes = [c for c in per_class if c.startswith("mine:")]
    if mine_classes:
        print(f"  your takes (balanced accuracy): rules {np.mean([np.mean(rules_per_class[c]) for c in mine_classes]):.1%}"
              f" -> combined {np.mean([np.mean(per_class[c]) for c in mine_classes]):.1%}")

    mean, scale = standardise(X)
    model = make().fit((X - mean) / scale, y)
    info = {"trained": time.strftime("%Y-%m-%d"), "model": name, "hands": dict(sorted(Counter(y).items())),
            "your_takes": len({r[1] for r in mine}), "min_none_confidence": min_none,
            "data": "HaGRID v2 train split (512 px), see tools/train_gesture_model.py",
            "out_of_fold_balanced_accuracy": {"rules": round(rules_score, 4), "combined": round(combined, 4)},
            "sklearn": sklearn.__version__}
    spec = export(model, mean, scale, classes, thresholds, info)
    # The numpy forward pass must reproduce scikit-learn exactly.
    numpy_proba = GestureModel(json.loads(json.dumps(spec))).probabilities(X[:500])
    assert np.abs(numpy_proba - model.predict_proba((X[:500] - mean) / scale)).max() < 1e-9
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, separators=(",", ":")))
    print(f"Wrote {output} ({output.stat().st_size / 1024:.0f} KB)")


def main():
    from gesture_model import DEFAULT_CLASSIFIER_PATH

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extract-only", action="store_true", help="only download photos and cache landmarks")
    parser.add_argument("--output", type=Path, default=DEFAULT_CLASSIFIER_PATH)
    parser.add_argument("--samples", type=Path, default=ROOT / "samples", help="your takes from tools/record_samples.py")
    # HaGRID's idle hands hang at people's sides, far from the camera. Tuned on those alone, the
    # gate hid 6 of 8 close-up open palms in the sample photos (0 of 8 at 0.8). Lower it only
    # once your own takes (especially "no gesture" and "Open Palm") show it is safe.
    parser.add_argument("--min-none-confidence", type=float, default=0.8,
                        help="lowest confidence at which a hand's label is hidden as 'no gesture' (default 0.8)")
    args = parser.parse_args()
    extract_landmarks()
    if not args.extract_only:
        train(args.output, samples=args.samples if args.samples.is_dir() else None, min_none=args.min_none_confidence)


if __name__ == "__main__":
    main()
