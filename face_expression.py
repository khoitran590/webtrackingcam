"""Facial expression recognition: which of seven basic expressions each face shows.

MediaPipe's face landmarker finds each face. The face is turned so its eyes are level,
cropped to a square around its landmarks, and classified by HSEmotion's EfficientNet-B0
("enet_b0_8_best_afew", A. Savchenko, trained on AffectNet; Apache-2.0,
https://github.com/sb-ai-lab/EmotiEffLib), run with OpenCV's DNN module.

The labels name the expression a face *shows*, not what the person feels: a polite smile
reads as Happy, and people show the same feeling in different ways.

The network also has a "contempt" class. It is AffectNet's least reliable class and no
photo set used here can measure it, so it is dropped and the other seven renormalised.

As published, the network calls about one resting face in five Angry (RAF-DB 20%, AffectNet 15%).
CLASS_OFFSETS shift its scores to correct that; see tools/fit_expression_offsets.py.
"""

import math
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from gesture_logic import HandTracker

FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
EXPRESSION_MODEL_URL = "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx"
EXPRESSION_MODEL_SHA256 = "7aa2ea31c1311f4f8aa9d3fdb085d418dd4e7a48c4b9ed41df8c044f91d0213f"

MODEL_CLASSES = ("anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")
LABELS = {"anger": "Angry", "disgust": "Disgusted", "fear": "Fearful", "happy": "Happy", "neutral": "Neutral",
          "sad": "Sad", "surprise": "Surprised"}
EXPRESSIONS = tuple(LABELS)  # the classes shown, in the order of every probability vector here
_KEEP = [MODEL_CLASSES.index(c) for c in EXPRESSIONS]
INPUT_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet statistics, as the network was trained
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# MediaPipe face-mesh eye corners (outer, inner) of the person's right and left eye.
RIGHT_EYE = (33, 133)
LEFT_EYE = (263, 362)

# Added to the network's scores (log-probabilities) per class, in EXPRESSIONS order. Fitted by
# tools/fit_expression_offsets.py to maximise mean per-expression accuracy on RAF-DB train and half of
# AffectNet together; offsets fitted to RAF-DB alone scored 8 points lower on AffectNet. On the photos
# not used for fitting: RAF-DB test 65.7% -> 70.6%, AffectNet 65.9% -> 65.7%, resting faces read as
# Angry 20% -> 8% (RAF-DB) and 15% -> 5% (AffectNet).
CLASS_OFFSETS = (-0.9, 0.15, -0.6, -0.1, 0.0, -0.3, 0.25)
CROP_MARGIN = 1.0  # crop side / landmark extent; 1.0 scored best of 0.8-1.4 on RAF-DB train
MIN_CONFIDENCE = 0.5  # smoothed probability an expression needs before it is shown
SMOOTHING_SECONDS = 0.3  # time constant of the moving average over a face's probabilities


def _region(connections):
    return tuple(sorted({i for c in connections for i in (c.start, c.end)}))


_MESH = mp.tasks.vision.FaceLandmarksConnections
# The parts of the face an expression is read from. A hand over most of one of them (a hand over
# the mouth, a palm over an eye) leaves the network guessing, so the face is reported as covered.
FACE_REGIONS = {"right eye": _region(_MESH.FACE_LANDMARKS_RIGHT_EYE), "left eye": _region(_MESH.FACE_LANDMARKS_LEFT_EYE),
                "mouth": _region(_MESH.FACE_LANDMARKS_LIPS)}
COVERED_SHARE = 0.5  # share of a region's landmarks under a hand that makes it covered
HAND_PAD = 0.15  # the hand's outline is widened by this many palm lengths: landmarks run down the finger middles
# A "hand" found mostly inside a face, with a palm shorter than this many face sides, is a phantom:
# on close-up face photos the hand model finds squashed hands on eyebrows, lips and noses (palms 0.02-0.3
# face sides), while real hands held in front of a face in HaGRID photos measured 0.35-0.71.
PHANTOM_MAX_PALM = 0.3


@dataclass
class FaceBox:
    """The square of the image the network sees: centre (x, y) in pixels, side in pixels,
    and angle: radians the eye line is turned from level (positive = clockwise on screen)."""

    centre: tuple
    side: float
    angle: float

    def _turn(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([[c, s], [-s, c]])  # turns image vectors by -angle, i.e. levels the face

    def corners(self):
        """The four corners in pixels: top-left, top-right, bottom-right, bottom-left of the face."""
        h = self.side / 2
        square = np.array([(-h, -h), (h, -h), (h, h), (-h, h)])
        return square @ self._turn() + self.centre

    def share_inside(self, points_px):
        """Share of ``points_px`` (x, y) that fall inside the box."""
        Q = (np.asarray(points_px, dtype=float)[:, :2] - self.centre) @ self._turn().T
        return float(np.mean(np.all(np.abs(Q) <= self.side / 2, axis=1)))

    def affine(self, size=INPUT_SIZE):
        """2x3 matrix taking image pixels to a size x size crop with the face upright."""
        k = size / self.side
        A = k * self._turn()
        return np.hstack([A, (size / 2 - A @ np.asarray(self.centre, float))[:, None]])


def face_box(points_px, margin=CROP_MARGIN, level=True):
    """The crop for one face from its landmarks in pixels (x, y[, z]).

    With ``level``, the box is turned with the face so the eyes come out level, and it is
    fitted to the landmarks in that turned frame, so it is equally snug at any tilt."""
    P = np.asarray(points_px, dtype=float)[:, :2]
    angle = 0.0
    if level:
        dx, dy = P[list(LEFT_EYE)].mean(0) - P[list(RIGHT_EYE)].mean(0)
        angle = math.atan2(dy, dx)
    box = FaceBox((0.0, 0.0), 1.0, angle)
    Q = P @ box._turn().T
    lo, hi = Q.min(0), Q.max(0)
    centre = (lo + hi) / 2 @ box._turn()
    return FaceBox(tuple(centre), float((hi - lo).max() * margin), angle)


def face_phantom(hand_px, boxes, max_palm=PHANTOM_MAX_PALM):
    """Whether a detected hand (21 (x, y) image points) is a phantom on a face: mostly inside
    one of the face ``boxes``, but far too small to be a real hand at that distance."""
    P = np.asarray(hand_px, dtype=float)[:, :2]
    palm = np.linalg.norm(P[9] - P[0])  # wrist to middle-finger knuckle
    return any(box.share_inside(P) >= 0.5 and palm < max_palm * box.side for box in boxes)


def covered_regions(face_px, hands_px, share=COVERED_SHARE, pad=HAND_PAD):
    """Names of the FACE_REGIONS that the hands (each 21 (x, y) image points) cover."""
    covered = []
    hulls = []
    for hand in hands_px:
        P = np.asarray(hand, dtype=np.float32)[:, :2]
        hulls.append((cv2.convexHull(P), pad * float(np.linalg.norm(P[9] - P[0]))))
    if not hulls:
        return covered
    for name, region in FACE_REGIONS.items():
        under = sum(any(cv2.pointPolygonTest(hull, (float(face_px[i][0]), float(face_px[i][1])), True) >= -margin
                        for hull, margin in hulls) for i in region)
        if under >= share * len(region):
            covered.append(name)
    return covered


def crop_face(image, box, size=INPUT_SIZE):
    """The ``size`` x ``size`` patch of ``image`` inside ``box``, face upright."""
    return cv2.warpAffine(image, box.affine(size), (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def softmax(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class ExpressionModel:
    """The HSEmotion network, run with OpenCV DNN. Takes RGB face crops, returns
    probabilities over EXPRESSIONS."""

    def __init__(self, path, offsets=CLASS_OFFSETS):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.offsets = np.zeros(len(EXPRESSIONS)) if offsets is None else np.asarray(offsets, dtype=float)

    def probabilities(self, faces_rgb):
        """(n, 7) probabilities for n INPUT_SIZE x INPUT_SIZE RGB crops (uint8)."""
        out = []
        for face in faces_rgb:  # the exported network takes one face at a time
            x = (np.asarray(face, np.float32) / 255.0 - _MEAN) / _STD
            self.net.setInput(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
            out.append(self.net.forward()[0, _KEEP])
        return softmax(np.array(out).reshape(-1, len(EXPRESSIONS)) + self.offsets)


def expression_label(probabilities, min_confidence=MIN_CONFIDENCE):
    """(label, probability) of the most likely expression; label is None below ``min_confidence``."""
    best = int(np.argmax(probabilities))
    p = float(probabilities[best])
    return (LABELS[EXPRESSIONS[best]] if p >= min_confidence else None), p


class ExpressionSmoother:
    """Exponential moving average of one face's probabilities, weighted by elapsed time so it
    behaves the same at any frame rate."""

    def __init__(self, time_constant=SMOOTHING_SECONDS):
        self.time_constant = time_constant
        self.value = self._t = None

    def __call__(self, probabilities, t):
        probabilities = np.asarray(probabilities, dtype=float)
        if self.value is None:
            self.value = probabilities
        elif t > self._t:
            alpha = 1.0 - math.exp(-(t - self._t) / self.time_constant)
            self.value = alpha * probabilities + (1 - alpha) * self.value
        else:
            return self.value
        self._t = t
        return self.value


class FaceTrack:
    def __init__(self, track_id, position, time_constant):
        self.id = track_id
        self.position = position
        self.missed = 0
        self.hits = 0
        self.velocity = (0.0, 0.0)
        self.labels = deque(maxlen=1)  # faces have no Left/Right label
        self.smoother = ExpressionSmoother(time_constant)


class FaceTracker(HandTracker):
    """Keeps a face's smoothing state across frames by matching it to the nearest face
    centre from the previous frames (same matching as hands)."""

    def __init__(self, max_jump=0.25, max_missed=10, time_constant=SMOOTHING_SECONDS):
        super().__init__(max_jump=max_jump, max_missed=max_missed)
        self.time_constant = time_constant

    def new_track(self, track_id, position):
        return FaceTrack(track_id, position, self.time_constant)


@dataclass
class FaceReading:
    track_id: int
    expression: str  # smoothed label shown on screen; None = no expression is likely enough
    confidence: float  # smoothed probability of the most likely expression
    probabilities: np.ndarray  # smoothed, in EXPRESSIONS order
    raw_probabilities: np.ndarray  # this frame's, in EXPRESSIONS order; None while covered
    box: FaceBox
    covered: tuple = ()  # FACE_REGIONS a hand covers; while any are, the expression is not read


def create_face_landmarker(model_path, num_faces=1, running_mode=None):
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode or vision.RunningMode.VIDEO,
        num_faces=num_faces,
    )
    return vision.FaceLandmarker.create_from_options(options)


class FacePipeline:
    """Runs the face landmarker on one frame and gives each face a smoothed expression.

    ``detect`` finds the faces and ``read`` classifies them, so hands found in between can be
    checked against the faces (face_phantom) and passed to ``read``: a face whose eyes or mouth
    a hand covers keeps its last expression, unshown, instead of being read through the hand."""

    def __init__(self, landmarker, model, tracker=None, min_confidence=MIN_CONFIDENCE, margin=CROP_MARGIN):
        self.landmarker = landmarker
        self.model = model
        self.tracker = tracker or FaceTracker()
        self.min_confidence = min_confidence
        self.margin = margin
        self.enabled = True
        self._last_timestamp_ms = -1
        self._frame = None  # (RGB frame, each face's landmarks in pixels, boxes) from detect

    def detect(self, frame_bgr, t_seconds):
        """Find the faces in a frame. Returns their FaceBoxes; call ``read`` next."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(int(t_seconds * 1000), self._last_timestamp_ms + 1)  # VIDEO mode needs increasing times
        self._last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        faces = [[(p.x * w, p.y * h) for p in face] for face in result.face_landmarks]
        boxes = [face_box(face, self.margin) for face in faces]
        self._frame = (rgb, faces, boxes)
        return boxes

    def read(self, t_seconds, hands_px=()):
        """Expressions of the faces ``detect`` found. ``hands_px``: this frame's hands, each 21 (x, y) points."""
        rgb, faces, boxes = self._frame
        h, w = rgb.shape[:2]
        tracks = self.tracker.update([(box.centre[0] / w, box.centre[1] / h) for box in boxes])
        covered = [tuple(covered_regions(face, hands_px)) for face in faces]
        visible = [i for i, c in enumerate(covered) if not c]
        raw = dict(zip(visible, self.model.probabilities([crop_face(rgb, boxes[i]) for i in visible]))) if visible else {}
        readings = []
        for i, (track, box) in enumerate(zip(tracks, boxes, strict=True)):
            if covered[i]:  # keep the last smoothed expression for when the hand moves away
                smoothed = track.smoother.value
                confidence = 0.0 if smoothed is None else float(np.max(smoothed))
                readings.append(FaceReading(track.id, None, confidence, smoothed, None, box, covered[i]))
                continue
            smoothed = track.smoother(raw[i], t_seconds)
            label, confidence = expression_label(smoothed, self.min_confidence)
            readings.append(FaceReading(track.id, label, confidence, smoothed, raw[i], box))
        return readings

    def process(self, frame_bgr, t_seconds, hands_px=()):
        """``detect`` and ``read`` in one step."""
        self.detect(frame_bgr, t_seconds)
        return self.read(t_seconds, hands_px)
