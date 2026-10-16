"""Facial expressions: crop geometry, smoothing and score offsets without models, then the real
face landmarker and expression network on sample photos (downloaded into tests/.cache/)."""

import math
import ssl
import urllib.request
from pathlib import Path

import certifi
import cv2
import numpy as np
import pytest

import face_expression as fe
from face_expression import (
    CLASS_OFFSETS,
    EXPRESSIONS,
    INPUT_SIZE,
    LEFT_EYE,
    MODEL_CLASSES,
    RIGHT_EYE,
    ExpressionModel,
    ExpressionSmoother,
    FacePipeline,
    FaceTracker,
    create_face_landmarker,
    FaceBox,
    covered_regions,
    expression_label,
    face_box,
    face_phantom,
)

CACHE = Path(__file__).resolve().parent / ".cache" / "images"
# MediaPipe's own face test photos: a broad smile, upright and turned 90 degrees.
PHOTOS = {"portrait": "https://storage.googleapis.com/mediapipe-assets/portrait.jpg",
          "portrait_rotated": "https://storage.googleapis.com/mediapipe-assets/portrait_rotated.jpg"}


def make_face(rng, angle=0.0, scale=1.0, shift=(0.0, 0.0)):
    """478 face-like points: eyes level at y=-0.2, the rest scattered in a face-sized oval,
    turned by ``angle`` radians (clockwise on screen), scaled and moved."""
    pts = np.column_stack([rng.uniform(-0.45, 0.45, 478), rng.uniform(-0.6, 0.6, 478)])
    for i, x in zip(RIGHT_EYE + LEFT_EYE, (-0.35, -0.12, 0.35, 0.12)):  # outer, inner corners
        pts[i] = (x, -0.2)
    c, s = math.cos(angle), math.sin(angle)
    return (pts @ np.array([[c, s], [-s, c]])) * scale * 200 + np.array([640, 360]) + shift


def to_crop(box, pts):
    M = box.affine()
    return np.asarray(pts) @ M[:, :2].T + M[:, 2]


def eye_centres(pts):
    pts = np.asarray(pts)
    return pts[list(RIGHT_EYE)].mean(0), pts[list(LEFT_EYE)].mean(0)


@pytest.mark.parametrize("degrees", [0, 15, -30, 90, 180, -135])
def test_crop_turns_the_eyes_level_with_the_right_eye_on_the_left(degrees):
    pts = make_face(np.random.default_rng(0), math.radians(degrees))
    box = face_box(pts)
    right, left = eye_centres(to_crop(box, pts))
    assert right[1] == pytest.approx(left[1], abs=1e-6)
    assert right[0] < left[0]
    assert math.degrees(box.angle) == pytest.approx(degrees, abs=1e-6)


@pytest.mark.parametrize("degrees, scale, shift", [(0, 1.0, (0, 0)), (40, 0.5, (100, -50)), (-90, 2.0, (-300, 80))])
def test_crop_is_the_same_at_any_tilt_size_and_position(degrees, scale, shift):
    rng = np.random.default_rng(1)
    upright = make_face(rng)
    moved = make_face(np.random.default_rng(1), math.radians(degrees), scale, shift)
    assert to_crop(face_box(moved), moved) == pytest.approx(to_crop(face_box(upright), upright), abs=1e-6)


def test_crop_fits_the_landmarks_and_corners_match_the_crop():
    pts = make_face(np.random.default_rng(2), math.radians(25))
    box = face_box(pts, margin=1.0)
    inside = to_crop(box, pts)
    assert inside.min() >= -1e-6 and inside.max() <= INPUT_SIZE + 1e-6
    assert min(inside.min(0)) == pytest.approx(0, abs=1e-6) or max(inside.max(0)) == pytest.approx(INPUT_SIZE)
    corners = to_crop(box, box.corners())
    assert corners == pytest.approx(np.array([(0, 0), (INPUT_SIZE, 0), (INPUT_SIZE, INPUT_SIZE), (0, INPUT_SIZE)]))


def test_unlevelled_crop_is_upright_and_square():
    pts = make_face(np.random.default_rng(3), math.radians(30))
    box = face_box(pts, level=False)
    assert box.angle == 0
    corners = box.corners()
    assert corners[1][0] - corners[0][0] == pytest.approx(box.side)
    assert corners[0][1] == pytest.approx(corners[1][1])


def test_label_needs_the_minimum_confidence():
    p = np.array([0.05, 0.05, 0.05, 0.55, 0.2, 0.05, 0.05])
    assert expression_label(p, 0.5) == ("Happy", pytest.approx(0.55))
    assert expression_label(p, 0.6) == (None, pytest.approx(0.55))


def test_labels_name_expressions_and_contempt_is_not_shown():
    assert set(fe.LABELS.values()) == {"Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"}
    assert "contempt" in MODEL_CLASSES and "contempt" not in EXPRESSIONS
    assert len(CLASS_OFFSETS) == len(EXPRESSIONS) and CLASS_OFFSETS[EXPRESSIONS.index("neutral")] == 0


class FakeNet:
    """Stands in for the ONNX network: returns fixed scores for the 8 model classes."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)
        self.inputs = []

    def setInput(self, blob):  # OpenCV's name
        self.inputs.append(blob)

    def forward(self):
        return self.logits[None]


def fake_model(logits, offsets=None):
    model = ExpressionModel.__new__(ExpressionModel)
    model.net = FakeNet(logits)
    model.offsets = np.zeros(len(EXPRESSIONS)) if offsets is None else np.asarray(offsets, dtype=float)
    return model


def test_contempt_is_dropped_and_the_rest_renormalised():
    logits = [0.0] * 8
    logits[MODEL_CLASSES.index("contempt")] = 10.0  # would win if it were kept
    p = fake_model(logits).probabilities([np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)])
    assert p.shape == (1, 7)
    assert p[0] == pytest.approx(np.full(7, 1 / 7))


def test_offsets_shift_the_scores():
    p = fake_model([0.0] * 8, CLASS_OFFSETS).probabilities([np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)])[0]
    expected = np.exp(CLASS_OFFSETS) / np.exp(CLASS_OFFSETS).sum()
    assert p == pytest.approx(expected)
    assert EXPRESSIONS[int(np.argmax(p))] == "surprise" and EXPRESSIONS[int(np.argmin(p))] == "anger"


def test_network_input_is_normalised_rgb_in_channel_order():
    model = fake_model([0.0] * 8)
    face = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    face[..., 0] = 255  # pure red
    model.probabilities([face])
    blob = model.net.inputs[0]
    assert blob.shape == (1, 3, INPUT_SIZE, INPUT_SIZE) and blob.dtype == np.float32
    assert blob[0, :, 0, 0] == pytest.approx([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225], rel=1e-5)


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_smoothing_follows_time_not_frame_rate(fps):
    smoother = ExpressionSmoother(time_constant=0.3)
    old, new = np.eye(7)[4], np.eye(7)[3]
    smoother(old, 0.0)
    for i in range(1, round(0.4 * fps) + 1):  # 0.4 s of the new expression
        value = smoother(new, i / fps)
    assert value[3] == pytest.approx(1 - math.exp(-0.4 / 0.3), abs=1e-9)


def test_smoothing_ignores_repeated_or_stale_times():
    smoother = ExpressionSmoother()
    first = smoother(np.eye(7)[0], 1.0)
    assert smoother(np.eye(7)[1], 1.0) == pytest.approx(first)
    assert smoother(np.eye(7)[1], 0.5) == pytest.approx(first)


def test_face_tracker_keeps_each_face_its_own_smoother():
    tracker = FaceTracker()
    a, b = tracker.update([(0.3, 0.5), (0.7, 0.5)])
    a.smoother(np.eye(7)[3], 0.0)
    a2, b2 = tracker.update([(0.32, 0.5), (0.68, 0.5)])
    assert (a2.id, b2.id) == (a.id, b.id)
    assert a2.smoother is a.smoother and b2.smoother.value is None
    (far,) = tracker.update([(0.0, 0.0)])  # a jump too far to be the same face
    assert far.id not in (a.id, b.id)


def flat_hand(centre, palm):
    """21 (x, y) points of an upright open hand: wrist at the bottom, palm length ``palm``."""
    pts = [(0.0, 0.0)] + [(dx, -palm * k) for dx in (-0.6, -0.3, 0.0, 0.3, 0.6) for k in (0.4, 0.8, 1.1, 1.4)]
    pts[9] = (0.0, -palm)  # the middle-finger knuckle sets the palm length
    return [(centre[0] + x * (palm if i else 1), centre[1] + y) for i, (x, y) in enumerate(pts)]


def test_face_phantom_is_a_small_hand_inside_a_face():
    face = FaceBox((500.0, 400.0), 200.0, 0.3)
    assert face_phantom(flat_hand((500, 420), 30), [face])  # squashed "hand" on the face
    assert not face_phantom(flat_hand((500, 480), 90), [face])  # a real hand held in front of it
    assert not face_phantom(flat_hand((900, 420), 30), [face])  # small, but nowhere near the face
    assert not face_phantom(flat_hand((500, 420), 30), [])


def face_with_regions():
    """478 points: each FACE_REGION as a small cluster, eyes at y=-40, mouth at y=+50, around (0, 0)."""
    pts = np.zeros((478, 2))
    for name, centre in {"right eye": (-40, -40), "left eye": (40, -40), "mouth": (0, 50)}.items():
        pts[list(fe.FACE_REGIONS[name])] = np.add(centre, np.linspace(-10, 10, len(fe.FACE_REGIONS[name]))[:, None])
    return pts + (500, 400)


def test_covered_regions_finds_what_the_hand_is_over():
    face = face_with_regions()
    assert covered_regions(face, []) == []
    assert covered_regions(face, [flat_hand((500, 520), 60)]) == ["mouth"]  # hand over the mouth
    assert covered_regions(face, [flat_hand((460, 40), 60)]) == []  # hand above the head
    assert set(covered_regions(face, [flat_hand((500, 560), 140)])) == set(fe.FACE_REGIONS)  # whole face


# --- The real models on photos ---------------------------------------------------------------


@pytest.fixture(scope="module")
def face_models():
    from hand_gesture_tracker import ensure_face_models

    try:
        return ensure_face_models()
    except RuntimeError as e:
        pytest.skip(f"face models unavailable: {e}")


def photo(name):
    path = CACHE / f"{name}.jpg"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(PHOTOS[name], context=context, timeout=30) as resp:
                path.write_bytes(resp.read())
        except OSError as e:
            pytest.skip(f"could not download {name}: {e}")
    image = cv2.imread(str(path))
    assert image is not None, f"unreadable image {path}"
    return image


def read_faces(face_models, frame, repeats=1, **kwargs):
    """Fresh landmarker per photo: VIDEO mode carries tracking state between frames."""
    with create_face_landmarker(face_models[0]) as landmarker:
        pipeline = FacePipeline(landmarker, ExpressionModel(face_models[1]), **kwargs)
        return [pipeline.process(frame, i / 30) for i in range(repeats)]


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("name, degrees", [("portrait", 0), ("portrait_rotated", 90)])
def test_smile_reads_as_happy_upright_turned_and_mirrored(face_models, name, degrees, mirrored):
    frame = photo(name)
    if mirrored:
        frame = cv2.flip(frame, 1)
    (faces,) = read_faces(face_models, frame)
    assert len(faces) == 1
    face = faces[0]
    assert face.expression == "Happy" and face.confidence > 0.9
    turn = -degrees if mirrored else degrees  # portrait_rotated is turned 90 degrees clockwise
    assert math.degrees(face.box.angle) == pytest.approx(turn, abs=10)
    h, w = frame.shape[:2]
    assert all(0 <= x <= w and 0 <= y <= h for x, y in face.box.corners())


def test_without_levelling_the_turned_smile_is_not_recognised(face_models):
    """Why the crop is levelled: the network was trained on upright faces."""
    import mediapipe as mp

    rgb = cv2.cvtColor(photo("portrait_rotated"), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    with create_face_landmarker(face_models[0]) as landmarker:
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), 0)
    points = [(p.x * w, p.y * h) for p in result.face_landmarks[0]]
    model = ExpressionModel(face_models[1])
    levelled, as_is = (model.probabilities([fe.crop_face(rgb, face_box(points, level=level))])[0]
                       for level in (True, False))
    happy = EXPRESSIONS.index("happy")
    assert levelled[happy] > 0.9
    assert as_is[happy] < 0.6


def test_no_face_gives_no_readings(face_models):
    (faces,) = read_faces(face_models, np.full((480, 640, 3), 128, np.uint8))
    assert faces == []


def test_repeated_frames_keep_one_track_and_a_steady_label(face_models):
    frames = read_faces(face_models, photo("portrait"), repeats=10)
    assert {f[0].track_id for f in frames} == {frames[0][0].track_id}
    assert all(f[0].expression == "Happy" for f in frames)
    assert frames[-1][0].probabilities == pytest.approx(frames[-1][0].raw_probabilities, abs=0.05)


def test_a_hand_over_the_mouth_holds_the_expression_until_it_moves_away(face_models):
    frame = photo("portrait")
    with create_face_landmarker(face_models[0]) as landmarker:
        pipeline = FacePipeline(landmarker, ExpressionModel(face_models[1]))
        [before] = pipeline.process(frame, 0.0)
        mouth = np.asarray(pipeline._frame[1][0])[list(fe.FACE_REGIONS["mouth"])].mean(0)
        palm = before.box.side * 0.5
        hand = flat_hand((mouth[0], mouth[1] + palm * 0.8), palm)
        [covered] = pipeline.process(frame, 1 / 30, [hand])
        [after] = pipeline.process(frame, 2 / 30)
    assert before.expression == "Happy"
    assert "mouth" in covered.covered and covered.expression is None and covered.raw_probabilities is None
    assert covered.track_id == before.track_id
    assert covered.probabilities == pytest.approx(before.probabilities)  # held, not read through the hand
    assert after.expression == "Happy" and not after.covered
