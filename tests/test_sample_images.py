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

from gesture_model import DEFAULT_CLASSIFIER_PATH, GestureModel
from hand_gesture_tracker import (
    GesturePipeline,
    avoid_labels,
    create_landmarker,
    draw_hand,
    draw_hud,
    ensure_model,
    label_box,
    label_origin,
)

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


@pytest.fixture(scope="session")
def classifier():
    if not DEFAULT_CLASSIFIER_PATH.exists():
        pytest.skip("no trained gesture model (tools/train_gesture_model.py)")
    return GestureModel.load()


def run_pipeline(model_path, frame, mirrored, repeats=1, classifier=None):
    """Fresh landmarker per image: VIDEO mode carries tracking state between frames."""
    with create_landmarker(model_path) as landmarker:
        pipeline = GesturePipeline(landmarker, mirrored=mirrored, model=classifier)
        for i in range(repeats):
            result = pipeline.process(frame, i / 30)
    return result.hands


@pytest.mark.parametrize("name", CASES)
def test_gestures_on_photos(model_path, name):
    _, expected, handedness = CASES[name]
    readings = run_pipeline(model_path, load_image(name), mirrored=False)
    assert sorted(r.gesture for r in readings) == expected
    if handedness:
        assert {r.handedness for r in readings} == {handedness}


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("mirrored", [False, True])
def test_gestures_on_photos_with_learned_model(model_path, classifier, name, mirrored):
    """Rules + model must still name every sample photo correctly, mirrored or not."""
    _, expected, _ = CASES[name]
    image = cv2.flip(load_image(name), 1) if mirrored else load_image(name)
    readings = run_pipeline(model_path, image, mirrored=mirrored, classifier=classifier)
    assert sorted(r.gesture for r in readings) == expected
    assert all(r.model_guess is not None for r in readings)


class StubModel:
    """Stands in for GestureModel: always gives the same answer."""

    def __init__(self, answer, confidence=0.99):
        self.answer, self.confidence = answer, confidence
        self.thresholds = {answer: 0.5}

    def predict(self, features):
        assert len(features) == 105
        return self.answer, self.confidence


def test_confident_no_gesture_hides_label_and_rules_label_is_kept(model_path):
    image = load_image("victory_a")
    [reading] = run_pipeline(model_path, image, mirrored=False, classifier=StubModel("none"))
    assert reading.gesture is None and reading.raw_gesture is None
    assert reading.rule_gesture == "Peace / Victory" and reading.model_guess == ("none", 0.99)
    canvas = image.copy()
    draw_hand(canvas, reading)  # just "Left"/"Right" above the hand
    assert not np.array_equal(canvas, image)


def test_model_can_be_switched_off(model_path):
    with create_landmarker(model_path) as landmarker:
        pipeline = GesturePipeline(landmarker, mirrored=False, model=StubModel("fist"))
        assert pipeline.use_model
        assert pipeline.process(load_image("victory_a"), 0.0).hands[0].raw_gesture == "Fist"
        pipeline.use_model = False
        reading = pipeline.process(load_image("victory_a"), 0.1).hands[0]
    assert reading.raw_gesture == reading.rule_gesture == "Peace / Victory" and reading.model_guess is None


@pytest.mark.parametrize("mode", [None, True, False])
def test_hud_draws_in_every_mode(mode):
    frame = np.zeros((720, 1280, 3), np.uint8)
    draw_hud(frame, 30.0, "Heart", mode)
    assert frame.any()


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
        history = [pipeline.process(image, i / 30).hands for i in range(15)]
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


def test_m_key_toggles_the_model(model_path, headless):
    app, keys = headless
    keys.extend([ord("m"), -1, ord("m"), ord("m"), ord("q")])
    with create_landmarker(model_path) as landmarker:
        pipeline = GesturePipeline(landmarker, model=StubModel("fist"))
        app.run(FakeCapture([]), pipeline, max_frames=100)
    assert pipeline.use_model is False  # toggled three times
    no_model = NoHands()
    keys.extend([ord("m"), ord("q")])
    app.run(FakeCapture([]), no_model, max_frames=100)
    assert no_model.use_model is False  # without a model, m does nothing


class NoHands:
    """A model-less pipeline that sees no hands, so the live loop runs without MediaPipe."""

    model, use_model = None, False

    def process(self, frame, t):
        from hand_gesture_tracker import FrameReading

        return FrameReading([])


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_live_loop_quits_on_q_and_esc(model_path, headless, key):
    app, keys = headless
    keys.extend([-1, -1, key])
    with create_landmarker(model_path) as landmarker:
        frames, _ = app.run(FakeCapture([]), GesturePipeline(landmarker), max_frames=100)
    assert frames == 3


@pytest.mark.parametrize("name", ["left_hands", "right_hands"])
@pytest.mark.parametrize("mirrored", [False, True])
def test_two_hand_gesture_on_photo(model_path, name, mirrored):
    """Two open hands, one upside down: a Double Open Palm, not a Frame (a real false positive once)."""
    image = load_image(name)
    frame = cv2.flip(image, 1) if mirrored else image
    with create_landmarker(model_path) as landmarker:
        result = GesturePipeline(landmarker, mirrored=mirrored).process(frame, 0.0)
    assert len(result.hands) == 2
    assert result.two_hand_gesture == "Double Open Palm"


def test_two_hand_gesture_absent_with_one_hand(model_path):
    with create_landmarker(model_path) as landmarker:
        result = GesturePipeline(landmarker).process(load_image("victory_a"), 0.0)
    assert len(result.hands) == 1 and result.two_hand_gesture is None


class ReplayLandmarker:
    """Stands in for the landmarker: returns pre-recorded results, one per frame."""

    def __init__(self, results):
        self.results = list(results)

    def detect_for_video(self, image, timestamp_ms):
        return self.results.pop(0)


def test_two_hand_label_is_voted_and_resets_when_hands_change(model_path):
    image = load_image("left_hands")
    with create_landmarker(model_path) as landmarker:
        both = landmarker.detect_for_video(mp_image(image), 0)
    one = type(both)(both.handedness[:1], both.hand_landmarks[:1], both.hand_world_landmarks[:1])
    none = type(both)([], [], [])
    script = [both, both, both, none, both, both, both, both, one, both]
    pipeline = GesturePipeline(ReplayLandmarker(script), mirrored=False)
    labels = [pipeline.process(image, i / 30).two_hand_gesture for i in range(len(script))]
    # Shown from the first frame; hidden while a hand is missing; after a gap the
    # tracks are the same hands again, so the label comes straight back.
    assert labels == ["Double Open Palm"] * 3 + [None] + ["Double Open Palm"] * 4 + [None, "Double Open Palm"]


def mp_image(bgr):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def test_two_hand_vote_smooths_flicker_and_restarts_for_new_pairs(model_path, monkeypatch):
    import hand_gesture_tracker as app

    image = load_image("left_hands")
    with create_landmarker(model_path) as landmarker:
        both = landmarker.detect_for_video(mp_image(image), 0)
    none = type(both)([], [], [])
    raw = iter(["Heart", None, "Heart", "Frame", "Heart", "Frame", "Frame", "Frame", "Heart", "Heart", "Heart", "Prayer"])
    monkeypatch.setattr(app, "classify_two_hands", lambda a, b: next(raw))
    script = [both] * 11 + [none] + [both]
    pipeline = GesturePipeline(ReplayLandmarker(script), mirrored=False)
    shown = [pipeline.process(image, i / 30).two_hand_gesture for i in range(len(script))]
    # 5-frame window: the single None and Frame glitches never show; Frame shows once it
    # holds 3 of 5 votes, and Heart wins back with 3 of 5. After the hands are lost, the
    # next pair starts a fresh vote, so the old Heart/Frame votes do not linger.
    assert shown == ["Heart"] * 6 + ["Frame"] * 4 + ["Heart"] + [None] + ["Prayer"]


def boxes_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize("y", [400, 60, 700])  # middle, against the HUD strip, at the bottom edge
def test_labels_of_nearby_hands_do_not_overlap(y):
    """Two-hand gestures put both hands (and so both labels) in the same place."""
    frame_size, size, placed = (1280, 720), (300, 20), []
    for _ in range(2):
        origin = avoid_labels((500, y), size, frame_size, placed, top=85)
        placed.append(label_box(origin, size))
    assert not boxes_overlap(*placed)
    for x0, y0, x1, y1 in placed:
        assert x0 >= 0 and x1 <= 1280 and y0 >= 85 - 6 and y1 <= 720


def test_first_label_is_not_moved():
    assert avoid_labels((500, 400), (300, 20), (1280, 720), [], top=85) == (500, 400)


def test_recorder_records_a_take_that_training_can_read(model_path, headless, tmp_path, monkeypatch):
    """Press '6' (Fist): 2 s countdown, 4 s of recording, then one file that train_gesture_model loads."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent / "tools"))
    import record_samples
    import train_gesture_model

    monkeypatch.setattr(record_samples.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(record_samples.cv2, "waitKey", lambda *a: keys.pop(0) if keys else -1)
    monkeypatch.setattr(record_samples, "window_visible", lambda: True)
    keys = [-1, ord("6")]
    ticks = iter(i * 0.5 for i in range(1000))  # 2 frames per second
    frame = load_image("fist")
    recorder = record_samples.Recorder(tmp_path)
    with create_landmarker(model_path) as landmarker:
        frames = record_samples.record(FakeCapture([(True, frame.copy()) for _ in range(20)]),
                                       GesturePipeline(landmarker), recorder, max_frames=20, clock=lambda: next(ticks))
    assert frames == 20 and len(recorder.saved) == 1
    rows = record_samples.load_takes(tmp_path)[recorder.saved[0]]
    assert len(rows) == 8  # 4 s at 2 frames per second, one hand per frame
    assert {r["class"] for r in rows} == {"fist"} and all(len(r["points_px"]) == len(r["world"]) == 21 for r in rows)
    assert record_samples.takes_per_class(tmp_path) == {"fist": 1}
    loaded = train_gesture_model.load_samples(tmp_path)
    assert len(loaded) == 8 and {r[0] for r in loaded} == {"mine:fist"} and len({r[1] for r in loaded}) == 1
    # What the recorder saved is what the app's rules see: the photo is a fist.
    hagrid_class, _, points_px, world, _ = loaded[0]
    assert train_gesture_model.rule_inputs(points_px, world)[0] == "Fist"
    assert train_gesture_model.is_correct("Fist", hagrid_class)


def test_recorder_ignores_new_keys_while_recording(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent / "tools"))
    import record_samples

    recorder = record_samples.Recorder(tmp_path)
    recorder.start("fist", 0.0)
    recorder.start("palm", 1.0)
    assert recorder.cls == "fist" and recorder.started == 0.0
