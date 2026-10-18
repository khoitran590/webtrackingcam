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

Your own face takes (tools/record_samples.py, samples/faces/) join the fit as a third set,
weighted like each dataset: the takes of each expression alternate between fitting and
checking, so the check is always on takes the offsets were not fitted to. Expressions you did
not record are left out of that set's score.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from face_expression import CLASS_OFFSETS, EXPRESSIONS  # noqa: E402

CACHE = ROOT / "tools" / ".cache" / "faces"
MINE = ROOT / "samples" / "faces"
FIXED = EXPRESSIONS.index("neutral")  # only differences between offsets matter, so one stays 0
GRID = np.arange(-3, 3.01, 0.05)


def load(path):
    """(true class index, log-probabilities) for the photos where a face was found."""
    d = np.load(path)
    found = ~np.isnan(d["probs"]).any(axis=1)
    truth = np.array([EXPRESSIONS.index(e) for e in d["truth"][found]])
    return truth, np.log(np.clip(d["probs"][found], 1e-12, 1.0))


def load_mine(folder):
    """(fitting, checking) sets from your takes: each expression's takes alternate between them."""
    takes = {}
    for path in sorted(Path(folder).glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            takes.setdefault(rows[0]["class"], []).append(rows)
    halves = ([], [])
    for paths in takes.values():
        for i, rows in enumerate(paths):
            halves[i % 2].extend(rows)

    def as_set(rows):
        truth = np.array([EXPRESSIONS.index(r["class"]) for r in rows], dtype=int)
        scores = np.array([r["scores"] for r in rows], dtype=float).reshape(-1, len(EXPRESSIONS))
        top = scores.max(axis=1, keepdims=True)
        return truth, scores - top - np.log(np.exp(scores - top).sum(axis=1, keepdims=True))

    return as_set(halves[0]), as_set(halves[1])


def per_expression(data, offsets):
    """Accuracy for each expression; NaN for expressions with no faces in ``data``."""
    truth, logp = data
    pred = (logp + offsets).argmax(axis=1)
    return np.array([np.mean(pred[truth == k] == k) if np.any(truth == k) else np.nan for k in range(len(EXPRESSIONS))])


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
                scores.append(np.mean([np.nanmean(per_expression(s, trial)) for s in sets]))
            offsets[k] = GRID[int(np.argmax(scores))]
    return offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rafdb-train", type=Path, default=CACHE / "rafdb_train.npz")
    parser.add_argument("--rafdb-test", type=Path, default=CACHE / "rafdb_test.npz")
    parser.add_argument("--affectnet", type=Path, default=CACHE / "affectnet_val.npz")
    parser.add_argument("--mine", type=Path, default=MINE, help="your face takes from tools/record_samples.py")
    parser.add_argument("--no-mine", action="store_true", help="fit to the datasets alone")
    args = parser.parse_args()

    raf_train, raf_test, affectnet = load(args.rafdb_train), load(args.rafdb_test), load(args.affectnet)
    order = np.random.default_rng(1).permutation(len(affectnet[0]))
    half = len(order) // 2
    aff_fit = tuple(a[order[:half]] for a in affectnet)
    aff_check = tuple(a[order[half:]] for a in affectnet)

    sets, report = [raf_train, aff_fit], [("RAF-DB train (fit)", raf_train), ("AffectNet half (fit)", aff_fit),
                                          ("RAF-DB test", raf_test), ("AffectNet other half", aff_check)]
    if not args.no_mine and args.mine.is_dir() and any(args.mine.glob("*.jsonl")):
        mine_fit, mine_check = load_mine(args.mine)
        print(f"Including your takes in {args.mine}: {len(mine_fit[0])} faces to fit, {len(mine_check[0])} to check")
        if len(mine_fit[0]):
            sets.append(mine_fit)
            report.insert(2, ("Your takes (fit)", mine_fit))
        if len(mine_check[0]):
            report.append(("Your takes (check)", mine_check))
        unchecked = sorted({EXPRESSIONS[k] for k in mine_fit[0]} - {EXPRESSIONS[k] for k in mine_check[0]})
        if unchecked:
            print(f"Only one take of {', '.join(unchecked)}: record another to check the refit on your face.")

    offsets = fit(sets)
    none, current = np.zeros(len(EXPRESSIONS)), np.asarray(CLASS_OFFSETS, dtype=float)
    print("CLASS_OFFSETS = (" + ", ".join(f"{v:g}" for v in offsets.round(2)) + ")  # " + ", ".join(EXPRESSIONS))
    print("Mean per-expression accuracy: no offsets / current CLASS_OFFSETS / new")
    for name, data in report:
        before, now, after = (per_expression(data, o) for o in (none, current, offsets))
        print(f"{name:22s} {np.nanmean(before):.1%} / {np.nanmean(now):.1%} / {np.nanmean(after):.1%}   "
              + " ".join(f"{e[:4]} {b:.0%}->{a:.0%}" for e, b, a in zip(EXPRESSIONS, now, after) if not np.isnan(b)))
        truth, logp = data
        resting = logp[truth == FIXED]
        if len(resting):
            shown = {EXPRESSIONS[k] for k in (resting + current).argmax(1)} | {EXPRESSIONS[k] for k in (resting + offsets).argmax(1)}
            print(f"{'':22s} resting faces read as: " + ", ".join(
                f"{e} {np.mean((resting + current).argmax(1) == EXPRESSIONS.index(e)):.0%}->"
                f"{np.mean((resting + offsets).argmax(1) == EXPRESSIONS.index(e)):.0%}" for e in sorted(shown) if e != "neutral"))


if __name__ == "__main__":
    main()
