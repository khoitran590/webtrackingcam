"""Run the real pipeline (MediaPipe model + gesture logic) on sample photos of hands.

The model and images are downloaded on first run into models/ and tests/.cache/.
"""

import urllib.request
from pathlib import Path

import certifi
import cv2
import numpy as np
import pytest
import ssl

from hand_gesture_tracker import GesturePipeline, create_landmarker, draw_hand, ensure_model, label_origin

CACHE = Path(__file__).resolve().parent / ".cache" / "images"
ASSETS = "https://storage.googleapis.com/mediapipe-assets/{}.jpg"
TASKS = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{}.jpg"

# image id -> (url, expected gestures sorted, expected handedness or None if unknown)
CASES = {
    "fist": (ASSETS.format("fist"), ["Fist"], None),
    "pointing_up_a": (ASSETS.format("pointing_up"), ["Pointing"], None),
    "pointing_up_b": (TASKS.format("pointing_up"), ["Pointing"], None),
    "thumb_up_a": (ASSETS.format("thumb_up"), ["Thumbs Up"], None),
    "thumbs_up_b": (TASKS.format("thumbs_up"), ["Thumbs Up"], None),
    "thumbs_down": (TASKS.format("thumbs_down"), ["Thumbs Down"], None),
    "victory_a": (ASSETS.format("victory"), ["Peace / Victory"], None),
    "victory_b": (TASKS.format("victory"), ["Peace / Victory"], None),
    # One upright hand plus one hanging upside down; the spec's y-test calls the second a fist.
    # Fingernails are visible, so these are backs of hands: left_hands really are left hands.
    "left_hands": (ASSETS.format("left_hands"), ["Open Palm", "Open Palm"], "Left"),
    "right_hands": (ASSETS.format("right_hands"), ["Open Palm", "Open Palm"], "Right"),
}


@pytest.fixture(scope="session")
def model_path():
    try:
        return ensure_model()
    except RuntimeError as e:
        pytest.skip(f"model unavailable: {e}")


def load_image(name):
    path = CACHE / f"{name}.jpg"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(CASES[name][0], context=context, timeout=30) as resp:
                path.write_bytes(resp.read())
        except OSError as e:
            pytest.skip(f"could not download {name}: {e}")
    image = cv2.imread(str(path))
    assert image is not None, f"unreadable image {path}"
    return image


def run_pipeline(model_path, frame, mirrored, repeats=1):
    """Fresh landmarker per image: VIDEO mode carries tracking state between frames."""
    with create_landmarker(model_path) as landmarker:
        pipeline = GesturePipeline(landmarker, mirrored=mirrored)
        for i in range(repeats):
            readings = pipeline.process(frame, i / 30)
    return readings


@pytest.mark.parametrize("name", CASES)
def test_gestures_on_photos(model_path, name):
    _, expected, handedness = CASES[name]
    readings = run_pipeline(model_path, load_image(name), mirrored=False)
    assert sorted(r.gesture for r in readings) == expected
    if handedness:
        assert {r.handedness for r in readings} == {handedness}


@pytest.mark.parametrize("name", CASES)
def test_mirrored_frame_gives_same_gestures_and_true_handedness(model_path, name):
    """The live app flips each frame. Gestures must not change, and the handedness label
    must still name the real hand (the Tasks model labels the mirrored look-alike)."""
    image = load_image(name)
    raw = run_pipeline(model_path, image, mirrored=False)
    flipped = run_pipeline(model_path, cv2.flip(image, 1), mirrored=True)
    assert sorted(r.gesture for r in flipped) == sorted(r.gesture for r in raw)
    assert sorted(r.handedness for r in flipped) == sorted(r.handedness for r in raw)


def test_repeated_frames_are_stable(model_path):
    """Feeding the same frame repeatedly through tracking + smoothing must not drift or flicker."""
    image = load_image("victory_a")
    with create_landmarker(model_path) as landmarker:
        pipeline = GesturePipeline(landmarker, mirrored=False)
        history = [pipeline.process(image, i / 30) for i in range(15)]
    assert all(len(r) == 1 for r in history)
    assert {r[0].gesture for r in history} == {"Peace / Victory"}
    assert {r[0].track_id for r in history} == {0}
    first, last = np.array(history[0][0].points_px), np.array(history[-1][0].points_px)
    assert np.abs(first - last).max() < 0.05 * image.shape[0]


def test_draw_hand_runs_on_real_reading(model_path):
    image = load_image("left_hands")
    readings = run_pipeline(model_path, image, mirrored=False)
    canvas = image.copy()
    for reading in readings:
        draw_hand(canvas, reading)
    assert not np.array_equal(canvas, image)


@pytest.mark.parametrize("ys", [(600, 715), (0, 120), (300, 400), (-50, 800)])
def test_label_always_on_screen(ys):
    """Spec drew labels at wrist_y + 30, which falls off a 720-px frame for low hands."""
    w, h, tw, th = 1280, 720, 300, 20
    pts = [(1250, ys[0]), (1270, ys[1])]
    x, y = label_origin(pts, (tw, th), (w, h), top=45)
    assert 0 <= x and x + tw <= w
    assert 45 <= y - th and y <= h  # on screen and clear of the HUD strip


class FakeCapture:
    """Stands in for cv2.VideoCapture: replays (ok, frame) pairs."""

    def __init__(self, reads):
        self.reads = list(reads)

    def read(self):
        return self.reads.pop(0) if self.reads else (True, self.last)

    @property
    def last(self):
        return np.zeros((72, 128, 3), np.uint8)


@pytest.fixture
def headless(monkeypatch):
    import hand_gesture_tracker as app

    keys = []
    monkeypatch.setattr(app.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda *a: keys.pop(0) if keys else -1)
    monkeypatch.setattr(app, "window_visible", lambda: True)
    monkeypatch.setattr(app.time, "sleep", lambda s: None)
    return app, keys


def test_live_loop_survives_warmup_drops_and_draws(model_path, headless):
    app, _ = headless
    frame = load_image("victory_a")
    cap = FakeCapture([(False, None)] * 5 + [(True, frame.copy()) for _ in range(5)])
    with create_landmarker(model_path) as landmarker:
        frames, _ = app.run(cap, GesturePipeline(landmarker), max_frames=5)
    assert frames == 5


def test_live_loop_gives_up_on_dead_camera(model_path, headless):
    app, _ = headless
    cap = FakeCapture([(False, None)] * (app.MAX_CONSECUTIVE_READ_FAILURES + 1))
    with create_landmarker(model_path) as landmarker, pytest.raises(RuntimeError, match="stopped delivering"):
        app.run(cap, GesturePipeline(landmarker), max_frames=5)


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_live_loop_quits_on_q_and_esc(model_path, headless, key):
    app, keys = headless
    keys.extend([-1, -1, key])
    with create_landmarker(model_path) as landmarker:
        frames, _ = app.run(FakeCapture([]), GesturePipeline(landmarker), max_frames=100)
    assert frames == 3
