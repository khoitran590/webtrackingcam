"""Measure facial expression accuracy on labelled face photos (RAF-DB or AffectNet).

Each photo goes through the app's own steps: MediaPipe face landmarks, the eye-level crop
(face_expression.face_box) and the expression network, without the smoothing over time.
A photo whose face is not found counts as a miss.

    uv pip install -r requirements-eval.txt
    .venv/bin/python tools/evaluate_faces.py                  # RAF-DB train: calibration photos
    .venv/bin/python tools/evaluate_faces.py --split test     # RAF-DB test: held-out photos
    .venv/bin/python tools/evaluate_faces.py --dataset affectnet

The expression network was trained on AffectNet and never saw RAF-DB, so both RAF-DB splits
are held out for it; the crop and the confidence cut-off were chosen on the train split only.
The AffectNet copy is a re-split of unknown origin and may contain the network's own training
photos, so its numbers can be optimistic.

Both datasets are licensed for non-commercial research only (RAF-DB: http://www.whdeng.cn/RAF/model1.html,
AffectNet: http://mohammadmahoor.com/affectnet/). They are read from unofficial copies on Hugging Face,
cached in tools/.cache/faces and never redistributed.
"""

import argparse
import shutil
import ssl
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import certifi
import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from face_expression import (  # noqa: E402
    CLASS_OFFSETS,
    CROP_MARGIN,
    EXPRESSIONS,
    MIN_CONFIDENCE,
    ExpressionModel,
    create_face_landmarker,
    crop_face,
    face_box,
)
from hand_gesture_tracker import ensure_face_models  # noqa: E402

HF = "https://huggingface.co/datasets/{}/resolve/main/data/{}"
# dataset -> split -> parquet file, and the dataset's label numbers -> expression (None = not used)
DATASETS = {
    "rafdb": ({"train": HF.format("elipaluma/RAF-DB_Kaggle", "train-00000-of-00001.parquet"),
               "test": HF.format("elipaluma/RAF-DB_Kaggle", "test-00000-of-00001.parquet")},
              {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "anger", 7: "neutral"}),
    "affectnet": ({"val": HF.format("Mauregato/affectnet_short", "val-00000-of-00001-15bd8d507ff5dd11.parquet")},
                  {0: "anger", 1: "surprise", 2: None, 3: "happy", 4: "neutral", 5: "fear", 6: "sad", 7: "disgust"}),
}
THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_photos(dataset, split, cache):
    """[(jpeg bytes, expression)] for one split, downloading the parquet file on first use."""
    import pyarrow.parquet as pq

    files, names = DATASETS[dataset]
    path = cache / dataset / f"{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {dataset} {split} to {path} ...", file=sys.stderr)
        context = ssl.create_default_context(cafile=certifi.where())
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(files[split], context=context, timeout=60) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(path)
    table = pq.read_table(path, columns=["image", "label"])
    return [(image["bytes"], names[label])
            for image, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist())
            if names[label] is not None]


def rotate(image, degrees):
    """Turn a photo by ``degrees`` (clockwise on screen) on a canvas big enough to keep all of it."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    size = (int(h * s + w * c), int(h * c + w * s))
    M[:, 2] += (size[0] - w) / 2, (size[1] - h) / 2
    return cv2.warpAffine(image, M, size, borderMode=cv2.BORDER_REPLICATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=DATASETS, default="rafdb")
    parser.add_argument("--split", help="rafdb: train (calibration, default) or test (held out); affectnet: val")
    parser.add_argument("--margin", type=float, default=CROP_MARGIN, help=f"crop margin (default {CROP_MARGIN})")
    parser.add_argument("--no-level", action="store_true", help="crop without turning the eyes level")
    parser.add_argument("--no-offsets", action="store_true", help="the network as published, without CLASS_OFFSETS")
    parser.add_argument("--rotate", type=float, default=0.0, help="turn every photo by this many degrees first")
    parser.add_argument("--pad", type=float, default=0.5,
                        help="border added around each photo, in photo sizes (default 0.5): these photos are "
                             "tight face crops, and MediaPipe misses many faces that fill the whole image")
    parser.add_argument("--limit", type=int, help="at most this many photos per expression")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                        help=f"the app's cut-off for showing a label (default {MIN_CONFIDENCE})")
    parser.add_argument("--cache", type=Path, default=ROOT / "tools" / ".cache" / "faces")
    parser.add_argument("--save", type=Path, help="also save every photo's probabilities (.npz)")
    args = parser.parse_args()
    split = args.split or next(iter(DATASETS[args.dataset][0]))

    photos = load_photos(args.dataset, split, args.cache)
    if args.limit:
        taken = Counter()
        photos = [p for p in photos if (taken.update([p[1]]) or taken[p[1]]) <= args.limit]
    landmarker_path, network_path = ensure_face_models()
    model = ExpressionModel(network_path, offsets=None if args.no_offsets else CLASS_OFFSETS)
    truth, probs = [], []  # probs row is None when no face was found
    with create_face_landmarker(landmarker_path, running_mode=mp.tasks.vision.RunningMode.IMAGE) as landmarker:
        for i, (jpeg, expression) in enumerate(photos):
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if args.pad:
                b = int(args.pad * max(image.shape[:2]))
                image = cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_REPLICATE)
            if args.rotate:
                image = rotate(image, args.rotate)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            truth.append(expression)
            if result.face_landmarks:
                box = face_box([(p.x * w, p.y * h) for p in result.face_landmarks[0]], args.margin,
                               level=not args.no_level)
                probs.append(model.probabilities([crop_face(rgb, box)])[0])
            else:
                probs.append(None)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1} / {len(photos)}", file=sys.stderr)
    if args.save:
        np.savez(args.save, truth=np.array(truth),
                 probs=np.array([p if p is not None else np.full(len(EXPRESSIONS), np.nan) for p in probs]))
    report(truth, probs, args)


def report(truth, probs, args):
    truth = np.array(truth)
    found = np.array([p is not None for p in probs])
    P = np.array([p if p is not None else np.zeros(len(EXPRESSIONS)) for p in probs])
    pred = np.array([EXPRESSIONS[i] for i in P.argmax(1)])
    conf = P.max(1)
    correct = found & (pred == truth)
    classes = [e for e in EXPRESSIONS if (truth == e).any()]

    level = "no levelling" if args.no_level else "eyes levelled"
    offsets = "no offsets" if args.no_offsets else "offsets"
    print(f"{len(truth)} photos, face found in {found.mean():.1%}  (margin {args.margin}, {level}, {offsets}, "
          f"turned {args.rotate:g} deg, pad {args.pad:g})")
    print(f"  {'':10s}{'photos':>8s}{'found':>8s}{'correct':>9s}{'of found':>10s}"
          f"{'shown':>8s}{'right':>8s}   most common wrong answers (faces found)")
    shown = found & (conf >= args.min_confidence)
    for e in classes:
        m = truth == e
        wrong = Counter(pred[m & found & ~correct]).most_common(3)
        print(f"  {e:10s}{m.sum():8d}{found[m].mean():8.0%}{correct[m].mean():9.0%}{correct[m & found].mean():10.0%}"
              f"{shown[m & found].mean():8.0%}{_rate(correct[m & shown]):>8s}   "
              + ", ".join(f"{w} {n / (m & found).sum():.0%}" for w, n in wrong))
    mean = np.mean([correct[truth == e].mean() for e in classes])
    mean_found = np.mean([correct[(truth == e) & found].mean() for e in classes])
    print(f"  mean per expression: {mean:.1%} of photos, {mean_found:.1%} of faces found;"
          f" overall {correct.mean():.1%} of photos")
    print("Confidence cut-off (faces found): share given a label, and how often that label is right")
    for t in THRESHOLDS:
        s = found & (conf >= t)
        per_class = [_rate(correct[s & (truth == e)]) for e in classes]
        marker = "  <- app" if abs(t - args.min_confidence) < 1e-9 else ""
        print(f"  >= {t:.1f}: shown {s.sum() / found.sum():5.1%}, right {_rate(correct[s]):>5s}"
              f"  (per expression: {' '.join(f'{e[:4]} {r}' for e, r in zip(classes, per_class))}){marker}")


def _rate(hits):
    return f"{hits.mean():.0%}" if len(hits) else "-"


if __name__ == "__main__":
    main()
