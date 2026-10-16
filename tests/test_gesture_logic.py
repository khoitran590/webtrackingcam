"""Unit tests for gesture_logic using synthetic hands (no camera, no model)."""

import math
import random

import numpy as np
import pytest

from gesture_logic import (
    PINCH_RATIO,
    GestureVote,
    HandPose,
    HandTracker,
    OneEuroFilter,
    classify_gesture,
    classify_two_hands,
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
    "Call Me": (True, False, False, False, True),
    "L Shape": (True, True, False, False, False),
    "3 Fingers Up": (False, True, True, True, False),
}


@pytest.mark.parametrize("expected,pose", POSES.items())
def test_synthetic_poses(expected, pose):
    world = make_hand(*pose)
    assert finger_states(world) == list(pose)
    assert classify_gesture(finger_states(world), pinch_ratio(world), "up") == expected


def test_ok_sign():
    world = make_ok_sign()
    assert pinch_ratio(world) < PINCH_RATIO
    assert classify_gesture(finger_states(world), pinch_ratio(world)) == "OK Sign"


@pytest.mark.parametrize("pose", list(POSES.values()) + ["ok"])
def test_finger_states_survive_any_3d_rotation_scale_and_mirror(pose):
    """The spec's y_tip < y_pip test breaks when the hand rotates; ours must not."""
    base = make_ok_sign() if pose == "ok" else make_hand(*pose)
    expected = finger_states(base)
    rng = np.random.default_rng(7)
    for _ in range(200):
        moved = transform(
            base, random_rotation(rng), scale=rng.uniform(0.5, 2.0),
            offset=rng.uniform(-1, 1, size=3), mirror=bool(rng.integers(2)),
        )
        assert finger_states(moved) == expected


def test_pinch_survives_in_plane_rotation_scale_and_mirror():
    """Pinch is measured in the image, so it is invariant to anything that keeps the
    hand facing the camera: in-plane rotation, zoom, position and mirroring."""
    base = make_ok_sign()
    expected = pinch_ratio(base)
    rng = np.random.default_rng(3)
    for _ in range(200):
        a = rng.uniform(0, 2 * math.pi)
        spin = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        moved = transform(base, spin, scale=rng.uniform(0.5, 4.0), offset=rng.uniform(-500, 500, size=3),
                          mirror=bool(rng.integers(2)))
        assert math.isclose(pinch_ratio(moved), expected, rel_tol=1e-9)


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


def test_tracker_confirms_a_hand_only_after_frames_in_a_row():
    tracker = HandTracker(confirm_frames=3)
    seen = [tracker.confirmed(t) for _ in range(3) for t in tracker.update([(0.2, 0.5)])]
    assert seen == [False, False, True]
    tracker.update([])  # a confirmed hand survives a missed frame
    assert [tracker.confirmed(t) for t in tracker.update([(0.2, 0.5)])] == [True]
    # A phantom that flickers in and out is dropped on every miss, so it is never confirmed.
    for _ in range(10):
        _, phantom = tracker.update([(0.2, 0.5), (0.8, 0.5)])
        assert not tracker.confirmed(phantom)
        tracker.update([(0.2, 0.5)])


def crossing(frames=9, speed=0.05):
    """Two hands moving towards each other, one slightly higher, and past each other; the hand
    in front hides the other at the crossing (frame 4). Yields (wrists, labels, true names)."""
    for f in range(frames):
        a, b = (0.3 + speed * f, 0.5), (0.7 - speed * f, 0.52)
        hands = [("a", a, "Right"), ("b", b, "Left")] if f != 4 else [("a", a, "Right")]
        yield [w for _, w, _ in hands], [l for _, _, l in hands], [n for n, _, _ in hands]


@pytest.mark.parametrize("use_labels", [True, False])
def test_tracker_keeps_identities_when_hands_cross(use_labels):
    """Nearest-wrist matching swapped them: right after the crossing each hand is nearest to
    where the other one was. Predicting from velocity, or the Left/Right labels, keeps them."""
    tracker, ids = HandTracker(), {}
    for wrists, labels, names in crossing():
        for name, track in zip(names, tracker.update(wrists, labels if use_labels else None)):
            assert ids.setdefault(name, track.id) == track.id


def test_tracker_label_does_not_override_a_clear_distance():
    """MediaPipe sometimes mislabels a hand; a wrong label must not pull it onto another track."""
    tracker = HandTracker()
    for _ in range(5):
        a, b = tracker.update([(0.3, 0.5), (0.7, 0.5)], ["Right", "Left"])
    a2, b2 = tracker.update([(0.3, 0.5), (0.7, 0.5)], ["Left", "Left"])
    assert (a2.id, b2.id) == (a.id, b.id)
    (only,) = tracker.update([(0.65, 0.5)], ["Right"])  # one hand hidden, the other mislabelled
    assert only.id == b.id


def test_tracker_equidistant_tracks_do_not_crash():
    tracker = HandTracker()
    tracker.update([(0.4, 0.5), (0.6, 0.5)])
    assert len(tracker.update([(0.5, 0.5)])) == 1


# --- Two-hand gestures -------------------------------------------------------------
# A 2D hand in image pixels (palm length 100), pointing up (-y), thumb on the +x side.
THUMB_2D = {True: [(25, -20), (45, -40), (70, -55), (95, -65)], False: [(25, -20), (40, -45), (35, -65), (15, -80)]}
FINGERS_2D = [  # MCP, then (extended, curled) PIP/DIP/TIP for index, middle, ring, pinky
    ((30, -95), [(33, -135), (35, -160), (37, -182)], [(33, -120), (30, -100), (28, -88)]),
    ((0, -100), [(0, -145), (0, -172), (0, -195)], [(0, -125), (-2, -105), (-3, -92)]),
    ((-25, -95), [(-27, -135), (-28, -160), (-29, -180)], [(-27, -118), (-26, -100), (-25, -88)]),
    ((-45, -85), [(-50, -115), (-53, -135), (-55, -152)], [(-48, -105), (-46, -92), (-44, -80)]),
]
OPEN, FIST, THUMB_ONLY, L_SHAPE = (True,) * 5, (False,) * 5, (True, False, False, False, False), (True, True, False, False, False)


def hand_2d(states, rotate=0.0, mirror=False, at=(0.0, 0.0)):
    """Template hand, mirrored, then rotated clockwise on screen by `rotate` degrees, then moved."""
    pts = [(0.0, 0.0)] + THUMB_2D[states[0]]
    for (mcp, ext, curl), up in zip(FINGERS_2D, states[1:]):
        pts += [mcp] + (ext if up else curl)
    if mirror:
        pts = [(-x, y) for x, y in pts]
    c, s = math.cos(math.radians(rotate)), math.sin(math.radians(rotate))
    return [(x * c - y * s + at[0], x * s + y * c + at[1]) for x, y in pts]


def pose(states, gesture="", **kwargs):
    return HandPose(hand_2d(states, **kwargs), list(states), gesture)


def mirrored(p):
    return HandPose([(-x, y) for x, y in p.points], p.fingers, p.gesture)


def heart():
    a = hand_2d(THUMB_ONLY, at=(-150, 150))
    a[4], a[8] = (-8, 40), (-8, -60)  # thumb tips meet at the bottom, index tips above them
    left = HandPose(a, list(THUMB_ONLY), "Thumb Out")
    return left, mirrored(left)


def frame():
    # Lower-left L, and an upper-right L rotated 180 degrees: a camera frame.
    return pose(L_SHAPE, "L Shape"), pose(L_SHAPE, "L Shape", rotate=180, at=(150, -250))


def time_out():
    level = pose(OPEN, "Open Palm", rotate=90, at=(0, 0))  # fingers pointing right
    cx = sum(level.points[i][0] for i in (0, 5, 9, 13, 17)) / 5
    cy = sum(level.points[i][1] for i in (0, 5, 9, 13, 17)) / 5
    upright = pose(OPEN, "Open Palm", at=(cx + 10, cy + 195))  # middle tip just under the flat palm
    return upright, level


def x_sign():
    left = pose(FIST, "Fist", rotate=-30, at=(-60, 0))  # leaning up and outwards
    return left, mirrored(left)


def prayer():
    left = pose(OPEN, "Open Palm", at=(-10, 0))
    return left, mirrored(left)


SCENES = {"Heart": heart, "Frame": frame, "Time Out": time_out, "X Sign": x_sign, "Prayer": prayer}


@pytest.mark.parametrize("expected", SCENES)
def test_two_hand_gestures(expected):
    a, b = SCENES[expected]()
    assert classify_two_hands(a, b) == expected
    assert classify_two_hands(b, a) == expected  # hand order must not matter


@pytest.mark.parametrize("expected", SCENES)
def test_two_hand_gestures_ignore_zoom_position_and_mirroring(expected):
    """The live view is mirrored and cameras differ in resolution."""
    a, b = SCENES[expected]()
    for scale, dx, dy, flip in [(0.3, 0, 0, False), (4.0, 900, 400, False), (1.0, 0, 0, True), (2.5, -300, 50, True)]:
        def move(p):
            return HandPose([((-x if flip else x) * scale + dx, y * scale + dy) for x, y in p.points], p.fingers, p.gesture)
        assert classify_two_hands(move(a), move(b)) == expected


def test_heart_needs_touching_tips_and_index_above_thumbs():
    a, b = heart()
    apart = HandPose([(x - 80, y) for x, y in a.points], a.fingers, a.gesture)
    assert classify_two_hands(apart, b) != "Heart"
    flipped = HandPose([(x, -y) for x, y in a.points], a.fingers, a.gesture)  # upside-down heart
    assert classify_two_hands(flipped, mirrored(flipped)) != "Heart"


def test_two_touching_fists_are_not_a_heart():
    a, b = heart()
    fists = [HandPose(h.points, list(FIST), "Fist") for h in (a, b)]
    assert classify_two_hands(*fists) == "Double Fist"


def test_side_by_side_l_shapes_are_double_not_frame():
    left = pose(L_SHAPE, "L Shape", mirror=True, at=(-200, 0))
    right = pose(L_SHAPE, "L Shape", at=(200, 0))
    assert classify_two_hands(left, right) == "Double L Shape"


def test_two_upright_open_hands_side_by_side_are_not_time_out_or_prayer():
    left = pose(OPEN, "Open Palm", mirror=True, at=(-200, 0))
    right = pose(OPEN, "Open Palm", at=(200, 0))
    assert classify_two_hands(left, right) == "Double Open Palm"


def test_upright_fists_are_not_x_sign():
    left = pose(FIST, "Fist", at=(-60, 0))
    assert classify_two_hands(left, mirrored(left)) == "Double Fist"


@pytest.mark.parametrize("gesture", ["Spider-Man / ILY", "Thumbs Up", "OK Sign", "Rock On", "Call Me"])
def test_double_named_gestures(gesture):
    left = pose(OPEN, gesture, mirror=True, at=(-400, 0))
    right = pose(OPEN, gesture, at=(400, 0))
    assert classify_two_hands(left, right) == f"Double {gesture}"


def test_no_double_for_different_gestures_or_finger_counts():
    assert classify_two_hands(pose(OPEN, "Spider-Man / ILY", at=(-400, 0)), pose(OPEN, "Rock On", at=(400, 0))) is None
    assert classify_two_hands(pose(OPEN, "2 Fingers Up", at=(-400, 0)), pose(OPEN, "2 Fingers Up", at=(400, 0))) is None


def test_two_hands_degenerate_does_not_crash():
    flat = HandPose([(0.0, 0.0)] * 21, [False] * 5, "Fist")
    assert classify_two_hands(flat, flat) is None


# Near misses: each scene breaks (mainly) one rule of a two-hand gesture, so every
# threshold in classify_two_hands is needed by at least one test.
def moved(p, dx=0.0, dy=0.0):
    return HandPose([(x + dx, y + dy) for x, y in p.points], p.fingers, p.gesture)


def with_point(p, i, xy):
    pts = list(p.points)
    pts[i] = xy
    return HandPose(pts, p.fingers, p.gesture)


def test_heart_needs_wrists_apart():
    a, b = heart()
    a = with_point(a, 0, (-40, 120))  # wrists nearly together: hands pressed, not a heart
    assert classify_two_hands(a, mirrored(a)) != "Heart"


def test_frame_needs_opposite_fingers():
    a, b = frame()
    sideways_index = with_point(b, 8, (b.points[5][0] + 100, b.points[5][1]))
    assert classify_two_hands(a, sideways_index) != "Frame"


def test_frame_needs_opposite_hands():
    a, b = frame()
    tilted = with_point(b, 0, (b.points[9][0] + 100, b.points[9][1]))  # wrist->knuckles now sideways
    assert classify_two_hands(a, tilted) != "Frame"


def test_frame_needs_hands_near_each_other():
    a, b = frame()
    assert classify_two_hands(a, moved(b, 2000, -2000)) != "Frame"


def test_time_out_needs_fingertips_at_the_palm():
    upright, level = time_out()
    assert classify_two_hands(moved(upright, 0, 400), level) != "Time Out"


def test_time_out_needs_one_upright_and_one_level_hand():
    upright, level = time_out()
    both_level = pose(OPEN, "Open Palm", rotate=90, at=upright.points[0])
    assert classify_two_hands(both_level, level) != "Time Out"
    # Two upright hands stacked, lower fingertips on the upper palm: not a T either.
    top = pose(OPEN, "Open Palm", at=(0, -300))
    centre = [sum(top.points[i][k] for i in (0, 5, 9, 13, 17)) / 5 for k in (0, 1)]
    bottom = pose(OPEN, "Open Palm", at=(centre[0], centre[1] + 195))
    assert classify_two_hands(bottom, top) != "Time Out"


def test_x_sign_needs_hands_pointing_up():
    left = pose(FIST, "Fist", rotate=-150, at=(-60, 0))  # leaning outwards but pointing down
    assert classify_two_hands(left, mirrored(left)) != "X Sign"


def test_x_sign_needs_hands_close_together():
    left = pose(FIST, "Fist", rotate=-30, at=(-400, 0))  # arms raised in a wide V, not crossed
    assert classify_two_hands(left, mirrored(left)) != "X Sign"


def test_prayer_needs_parallel_upright_hands():
    a = pose(OPEN, "Open Palm", rotate=20, at=(-67, 0))  # slight V, fingertips touching
    assert classify_two_hands(a, mirrored(a)) != "Prayer"
    down = pose(OPEN, "Open Palm", rotate=180, at=(-10, 0))
    assert classify_two_hands(down, mirrored(down)) != "Prayer"


def test_prayer_needs_hands_pressed_together():
    a = pose(OPEN, "Open Palm", at=(-10, 0))
    assert classify_two_hands(a, moved(mirrored(a), 0, 120)) != "Prayer"  # one hand lower


def test_open_hands_rotated_apart_are_not_a_frame():
    """Found on a real photo: an upright and an upside-down open hand read as Frame."""
    a = pose(OPEN, "Open Palm")
    b = pose(OPEN, "Open Palm", rotate=180, at=(150, -250))
    assert classify_two_hands(a, b) != "Frame"


def test_hands_without_gesture_names_are_not_doubled():
    assert classify_two_hands(pose(OPEN, "", at=(-400, 0)), pose(OPEN, "", at=(400, 0))) is None
