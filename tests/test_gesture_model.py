"""Unit tests for gesture_model: hand normalisation, the numpy forward pass, and how the
model's answer is combined with the rules. No camera, no MediaPipe."""

import json
import math

import numpy as np
import pytest

from gesture_model import DEFAULT_CLASSIFIER_PATH, FEATURE_VERSION, GestureModel, combine, hand_features

RNG = np.random.default_rng(7)
# An arbitrary but fixed hand: 21 image points (pixels) and world points (metres).
POINTS = RNG.uniform(100, 400, size=(21, 2))
WORLD = RNG.uniform(-0.08, 0.08, size=(21, 3))


def turn(points, world, degrees, scale=1.0, shift=(0.0, 0.0)):
    a = math.radians(degrees)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = (points - points[0]) @ r.T * scale + points[0] + shift
    w = world.copy()
    w[:, :2] = world[:, :2] @ r.T
    return p, w


@pytest.mark.parametrize("degrees", [0, 13, 90, 180, 271])
@pytest.mark.parametrize("scale", [0.2, 1.0, 5.0])
def test_features_ignore_rotation_size_and_position(degrees, scale):
    base = hand_features(POINTS, WORLD, "Right")
    moved = hand_features(*turn(POINTS, WORLD, degrees, scale, (-300, 950)), "Right")
    assert np.allclose(moved, base, atol=1e-9)


def test_features_turn_the_hand_upright():
    f = hand_features(POINTS, WORLD, "Right")
    assert np.allclose(f[0:2], 0) and np.allclose(f[18:20], (0, -1))  # wrist at origin, middle knuckle straight up


def test_a_mirrored_left_hand_looks_like_the_right_hand():
    """Mirroring and swapping the label gives the same features, so the model sees every hand as a right hand."""
    mirrored_points, mirrored_world = POINTS * (-1, 1), WORLD * (-1, 1, 1)
    assert np.allclose(hand_features(mirrored_points, mirrored_world, "Left"), hand_features(POINTS, WORLD, "Right"))
    assert not np.allclose(hand_features(mirrored_points, mirrored_world, "Right"), hand_features(POINTS, WORLD, "Right"))


def tiny_spec(**changes):
    rng = np.random.default_rng(0)
    spec = {"feature_version": FEATURE_VERSION, "classes": ["fist", "none", "palm"],
            "mean": rng.normal(size=4).tolist(), "scale": rng.uniform(0.5, 2, size=4).tolist(),
            "layers": [{"weights": rng.normal(size=(4, 5)).tolist(), "bias": rng.normal(size=5).tolist()},
                       {"weights": rng.normal(size=(5, 3)).tolist(), "bias": rng.normal(size=3).tolist()}],
            "thresholds": {"fist": 0.6, "none": 0.7}}
    return {**spec, **changes}


def test_forward_pass_matches_a_hand_computation():
    spec = tiny_spec()
    model = GestureModel(spec)
    x = np.array([0.3, -1.0, 2.0, 0.5])
    h = np.maximum(0, ((x - spec["mean"]) / spec["scale"]) @ np.array(spec["layers"][0]["weights"]) + spec["layers"][0]["bias"])
    logits = h @ np.array(spec["layers"][1]["weights"]) + spec["layers"][1]["bias"]
    expected = np.exp(logits) / np.exp(logits).sum()
    assert np.allclose(model.probabilities(x), expected)
    assert model.predict(x) == (spec["classes"][int(expected.argmax())], pytest.approx(expected.max()))
    batch = model.probabilities(np.stack([x, x * 2]))
    assert batch.shape == (2, 3) and np.allclose(batch.sum(axis=1), 1)


def test_missing_thresholds_mean_never_trust_that_class():
    assert GestureModel(tiny_spec()).thresholds == {"fist": 0.6, "none": 0.7, "palm": 1.0}


def test_model_from_other_feature_version_is_refused():
    with pytest.raises(ValueError, match="features"):
        GestureModel(tiny_spec(feature_version=FEATURE_VERSION + 1))


T = {"none": 0.7, "fist": 0.6, "thumbs": 0.5, "rock": 0.5, "other": 0.5, "ok": 0.5}
FIST, THUMB_ONLY, ROCK_THUMB_IN, ROCK_THUMB_OUT = [False] * 5, [True] + [False] * 4, [False, True, False, False, True], [True, True, False, False, True]


def test_rules_decide_when_the_model_is_unsure():
    assert combine("Open Palm", [True] * 5, "up", "none", 0.69, T) == "Open Palm"
    assert combine("3 Fingers Up", FIST, "up", "fist", 0.59, T) == "3 Fingers Up"


def test_confident_no_gesture_hides_the_label():
    assert combine("Open Palm", [True] * 5, "up", "none", 0.7, T) is None


def test_confident_shape_overrides_the_rules():
    assert combine("3 Fingers Up", FIST, "up", "fist", 0.9, T) == "Fist"
    assert combine("Open Palm", [True] * 5, "up", "ok", 0.9, T) == "OK Sign"


def test_other_shapes_keep_finger_counts_but_get_no_gesture_name():
    assert combine("4 Fingers Up", [False] + [True] * 4, "up", "other", 0.99, T) == "4 Fingers Up"
    assert combine("Thumbs Up", THUMB_ONLY, "up", "other", 0.99, T) is None  # a pen grip, not a thumbs up
    assert combine("Thumbs Up", THUMB_ONLY, "up", "other", 0.49, T) == "Thumbs Up"  # unsure: rules decide


@pytest.mark.parametrize("direction, label", [("up", "Thumbs Up"), ("down", "Thumbs Down"), ("side", "Thumb Out")])
def test_rules_pick_the_thumb_direction(direction, label):
    """Turning hands upright makes thumbs up and down look alike, so the model has one 'thumbs' class."""
    assert combine("Fist", FIST, direction, "thumbs", 0.9, T) == label


def test_rules_pick_rock_on_or_ily_from_the_thumb():
    assert combine("2 Fingers Up", ROCK_THUMB_IN, "up", "rock", 0.9, T) == "Rock On"
    assert combine("3 Fingers Up", ROCK_THUMB_OUT, "up", "rock", 0.9, T) == "Spider-Man / ILY"


@pytest.mark.skipif(not DEFAULT_CLASSIFIER_PATH.exists(), reason="no trained model")
def test_shipped_model_loads_and_is_consistent():
    spec = json.loads(DEFAULT_CLASSIFIER_PATH.read_text())
    model = GestureModel(spec)
    assert set(model.classes) >= {"none", "other", "thumbs", "rock", "ok", "fist", "palm", "peace", "one", "call", "l_shape"}
    assert len(model.mean) == len(hand_features(POINTS, WORLD, "Right"))
    assert all(0 <= t <= 1.01 for t in model.thresholds.values())
    p = model.probabilities(hand_features(POINTS, WORLD, "Right"))
    assert p.shape == (len(model.classes),) and math.isclose(p.sum(), 1.0)
