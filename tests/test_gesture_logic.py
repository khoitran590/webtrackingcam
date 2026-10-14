"""Unit tests for gesture_logic using synthetic hands (no camera, no model)."""

import math
import random

import numpy as np
import pytest

from gesture_logic import (
    GestureVote,
    HandTracker,
    OneEuroFilter,
    classify_gesture,
    finger_states,
    pinch_ratio,
    thumb_direction,
)

# Synthetic right hand in metres: wrist at the origin, fingers along +y, palm facing -z.
MCPS = {"index": (0.02, 0.09, 0.0), "middle": (0.0, 0.09, 0.0), "ring": (-0.017, 0.087, 0.0), "pinky": (-0.03, 0.08, 0.0)}
EXTENDED = [(0, 0.04, 0), (0, 0.025, 0), (0, 0.02, 0)]  # PIP, DIP, TIP offsets, chained
CURLED = [(0, 0.01, -0.035), (0, -0.02, -0.012), (0, -0.015, 0.01)]
THUMB_OUT = [(0.025, 0.025, 0), (0.045, 0.045, 0), (0.06, 0.07, 0), (0.07, 0.095, 0)]
THUMB_TUCKED = [(0.025, 0.025, 0), (0.035, 0.045, -0.01), (0.03, 0.06, -0.015), (0.0, 0.06, -0.025)]


def make_hand(thumb_out, index, middle, ring, pinky):
    pts = [(0.0, 0.0, 0.0)] + (THUMB_OUT if thumb_out else THUMB_TUCKED)
    for name, up in zip(MCPS, (index, middle, ring, pinky)):
        p = MCPS[name]
        pts.append(p)
        for step in EXTENDED if up else CURLED:
            p = tuple(a + b for a, b in zip(p, step))
            pts.append(p)
    assert len(pts) == 21
    return pts


def make_ok_sign():
    pts = make_hand(True, False, True, True, True)
    pts[4] = (0.04, 0.10, -0.03)  # thumb tip in front of the palm
    pts[6], pts[7], pts[8] = (0.02, 0.12, -0.02), (0.03, 0.115, -0.03), (0.04, 0.105, -0.032)
    return pts


def transform(points, rotation, scale=1.0, offset=(0, 0, 0), mirror=False):
    arr = np.array(points)
    if mirror:
        arr[:, 0] *= -1
    return [tuple(p) for p in (arr @ rotation.T) * scale + np.array(offset)]


def random_rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


POSES = {
    "Fist": (False, False, False, False, False),
    "Open Palm": (True, True, True, True, True),
    "Thumbs Up": (True, False, False, False, False),
    "Pointing": (False, True, False, False, False),
    "Peace / Victory": (False, True, True, False, False),
    "Rock On": (False, True, False, False, True),
    "Spider-Man / ILY": (True, True, False, False, True),
    "3 Fingers Up": (False, True, True, True, False),
}


@pytest.mark.parametrize("expected,pose", POSES.items())
def test_synthetic_poses(expected, pose):
    world = make_hand(*pose)
    assert finger_states(world) == list(pose)
    assert classify_gesture(finger_states(world), pinch_ratio(world), "up") == expected


def test_ok_sign():
    world = make_ok_sign()
    assert pinch_ratio(world) < 0.3
    assert classify_gesture(finger_states(world), pinch_ratio(world)) == "OK Sign"


@pytest.mark.parametrize("pose", list(POSES.values()) + ["ok"])
def test_rotation_scale_and_mirror_invariance(pose):
    """The spec's y_tip < y_pip test breaks when the hand rotates; ours must not."""
    base = make_ok_sign() if pose == "ok" else make_hand(*pose)
    expected_states, expected_pinch = finger_states(base), pinch_ratio(base)
    rng = np.random.default_rng(7)
    for _ in range(200):
        moved = transform(
            base, random_rotation(rng), scale=rng.uniform(0.5, 2.0),
            offset=rng.uniform(-1, 1, size=3), mirror=bool(rng.integers(2)),
        )
        assert finger_states(moved) == expected_states
        assert math.isclose(pinch_ratio(moved), expected_pinch, rel_tol=1e-9)


@pytest.mark.parametrize("tip,expected", [
    ((100, 20), "up"), ((130, 40), "up"),        # vertical / 37 degrees off vertical
    ((100, 180), "down"), ((70, 160), "down"),
    ((170, 100), "side"), ((30, 70), "side"),    # horizontal / 67 degrees off vertical
    ((100, 100), "side"),                         # zero-length thumb
])
def test_thumb_direction(tip, expected):
    points = [(0, 0)] * 21
    points[2], points[4] = (100, 100), tip
    assert thumb_direction(points) == expected


def test_thumb_only_variants_and_fallback_grammar():
    thumb_only = [True, False, False, False, False]
    assert classify_gesture(thumb_only, 1.0, "up") == "Thumbs Up"
    assert classify_gesture(thumb_only, 1.0, "down") == "Thumbs Down"
    assert classify_gesture(thumb_only, 1.0, "side") == "Thumb Out"
    assert classify_gesture([False, False, True, False, False], 1.0) == "1 Finger Up"
    assert classify_gesture([False, False, True, True, False], 1.0) == "2 Fingers Up"


def test_pinch_without_extended_fingers_is_not_ok_sign():
    assert classify_gesture([False] * 5, 0.1) == "Fist"


def test_degenerate_hand_does_not_crash():
    flat = [(0.0, 0.0, 0.0)] * 21
    assert finger_states(flat) == [False] * 5
    assert pinch_ratio(flat) == math.inf


def test_vote_suppresses_single_frame_flicker():
    vote = GestureVote(5)
    shown = [vote.update(l) for l in "AAABAABAA"]
    assert shown == list("AAAAAAAAA")


def test_vote_switches_once_outvoted_and_holds_on_ties():
    vote = GestureVote(5)
    shown = [vote.update(l) for l in "AAABBB"]
    assert shown == list("AAAAAB")
    tie = GestureVote(4)
    assert [tie.update(l) for l in "AABB"] == list("AAAA")


def test_one_euro_constant_and_noise():
    f = OneEuroFilter(min_cutoff=1.0, beta=10.0)
    assert all(f(0.5, i / 30) == 0.5 for i in range(30))

    rng = random.Random(1)
    noisy = [0.5 + rng.gauss(0, 0.01) for _ in range(300)]
    f = OneEuroFilter(min_cutoff=1.0, beta=10.0)
    out = [f(v, i / 30) for i, v in enumerate(noisy)]
    assert np.std(out[30:]) < 0.5 * np.std(noisy[30:])


def test_one_euro_follows_fast_motion_and_ignores_stale_time():
    f = OneEuroFilter(min_cutoff=1.0, beta=10.0)
    for i in range(10):
        f(0.0, i / 30)
    out = [f(1.0, (10 + i) / 30) for i in range(10)]
    assert out[-1] > 0.95  # caught up with the jump in a third of a second
    assert f(5.0, 19 / 30) == out[-1]  # timestamp did not advance: value unchanged


def test_tracker_keeps_ids_and_expires_lost_hands():
    tracker = HandTracker(max_jump=0.25, max_missed=2)
    a, b = tracker.update([(0.2, 0.5), (0.8, 0.5)])
    assert (a.id, b.id) == (0, 1)
    # Hands move a little and arrive in the opposite detection order.
    b2, a2 = tracker.update([(0.78, 0.52), (0.22, 0.49)])
    assert (a2.id, b2.id) == (0, 1)
    # Right hand drops out for two frames, then returns: same track.
    tracker.update([(0.22, 0.49)])
    tracker.update([(0.22, 0.49)])
    assert [t.id for t in tracker.update([(0.22, 0.49), (0.79, 0.5)])] == [0, 1]
    # Gone for three frames: expired, so it comes back as a new hand.
    for _ in range(3):
        tracker.update([(0.22, 0.49)])
    assert [t.id for t in tracker.update([(0.22, 0.49), (0.79, 0.5)])] == [0, 2]


def test_tracker_equidistant_tracks_do_not_crash():
    tracker = HandTracker()
    tracker.update([(0.4, 0.5), (0.6, 0.5)])
    assert len(tracker.update([(0.5, 0.5)])) == 1
