"""Fit face_expression.CLASS_OFFSETS: per-expression shifts of the network's scores.

Offsets are chosen to maximise mean per-expression accuracy on RAF-DB train and on half of the
AffectNet photos together, and are checked on RAF-DB test and the other AffectNet half. Fitting
to RAF-DB alone gave 72% on RAF-DB test but cost 8 points on AffectNet: the two datasets draw
the line between Surprised and Fearful differently, so only shifts that suit both are kept.

First save the network's probabilities without offsets (faces not found are skipped here):

    .venv/bin/python tools/evaluate_faces.py --no-offsets --save tools/.cache/faces/rafdb_train.npz
    .venv/bin/python tools/evaluate_faces.py --no-offsets --split test --save tools/.cache/faces/rafdb_test.npz
    .venv/bin/python tools/evaluate_faces.py --no-offsets --dataset affectnet --save tools/.cache/faces/affectnet_val.npz
    .venv/bin/python tools/fit_expression_offsets.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from face_expression import EXPRESSIONS  # noqa: E402

CACHE = ROOT / "tools" / ".cache" / "faces"
FIXED = EXPRESSIONS.index("neutral")  # only differences between offsets matter, so one stays 0
GRID = np.arange(-3, 3.01, 0.05)


def load(path):
    """(true class index, log-probabilities) for the photos where a face was found."""
    d = np.load(path)
    found = ~np.isnan(d["probs"]).any(axis=1)
    truth = np.array([EXPRESSIONS.index(e) for e in d["truth"][found]])
    return truth, np.log(np.clip(d["probs"][found], 1e-12, 1.0))


def per_expression(data, offsets):
    truth, logp = data
    pred = (logp + offsets).argmax(axis=1)
    return np.array([np.mean(pred[truth == k] == k) for k in range(len(EXPRESSIONS))])


def fit(sets, passes=6):
    """Coordinate ascent on the mean over ``sets`` of mean per-expression accuracy."""
    offsets = np.zeros(len(EXPRESSIONS))
    for _ in range(passes):
        for k in range(len(EXPRESSIONS)):
            if k == FIXED:
                continue
            scores = []
            for v in GRID:
                trial = offsets.copy()
                trial[k] = v
                scores.append(np.mean([per_expression(s, trial).mean() for s in sets]))
            offsets[k] = GRID[int(np.argmax(scores))]
    return offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rafdb-train", type=Path, default=CACHE / "rafdb_train.npz")
    parser.add_argument("--rafdb-test", type=Path, default=CACHE / "rafdb_test.npz")
    parser.add_argument("--affectnet", type=Path, default=CACHE / "affectnet_val.npz")
    args = parser.parse_args()

    raf_train, raf_test, affectnet = load(args.rafdb_train), load(args.rafdb_test), load(args.affectnet)
    order = np.random.default_rng(1).permutation(len(affectnet[0]))
    half = len(order) // 2
    aff_fit = tuple(a[order[:half]] for a in affectnet)
    aff_check = tuple(a[order[half:]] for a in affectnet)

    offsets = fit([raf_train, aff_fit])
    none = np.zeros(len(EXPRESSIONS))
    print("CLASS_OFFSETS = (" + ", ".join(f"{v:g}" for v in offsets.round(2)) + ")  # " + ", ".join(EXPRESSIONS))
    for name, data in [("RAF-DB train (fit)", raf_train), ("AffectNet half (fit)", aff_fit),
                       ("RAF-DB test", raf_test), ("AffectNet other half", aff_check)]:
        before, after = per_expression(data, none), per_expression(data, offsets)
        print(f"{name:22s} {before.mean():.1%} -> {after.mean():.1%}   "
              + " ".join(f"{e[:4]} {b:.0%}->{a:.0%}" for e, b, a in zip(EXPRESSIONS, before, after)))
        truth, logp = data
        neutral = truth == FIXED
        angry = EXPRESSIONS.index("anger")
        print(f"{'':22s} resting faces read as Angry: {np.mean(logp[neutral].argmax(1) == angry):.0%} -> "
              f"{np.mean((logp[neutral] + offsets).argmax(1) == angry):.0%}")


if __name__ == "__main__":
    main()
