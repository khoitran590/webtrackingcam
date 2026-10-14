"""Gesture classification and smoothing, independent of the camera, OpenCV and MediaPipe.

A hand is a sequence of 21 points in MediaPipe's landmark order.
Finger states are computed from MediaPipe's *world* landmarks (metric 3D, centred on
the hand). They only compare 3D distances, so they do not depend on hand rotation,
palm-vs-back orientation, handedness, or distance from the camera.
Everything that relates two points that may touch (the OK-sign pinch) or two hands
to each other uses image points in pixels, because world landmarks are centred on
each hand separately and their depth estimates are too noisy for fingertip contact.

Thresholds were calibrated on 1,700 photos from the HaGRID v2 test split (100 per
class) and checked on a held-out 1,700 from other shards; see tools/evaluate_hagrid.py.
"""

import math
from collections import Counter, deque, namedtuple

WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP = 13
PINKY_MCP = 17
FINGER_TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky
FINGER_PIPS = (6, 10, 14, 18)
PALM_POINTS = (0, 5, 9, 13, 17)

# Wrist->tip over wrist->PIP (3D) for index, middle, ring, pinky: 97-99% balanced
# accuracy per finger on ~3,600 HaGRID fingers. The pinky, the shortest finger, is
# often left slightly bent when "extended" (e.g. Call Me), so its cutoff is lower.
FINGER_EXTENSION_RATIOS = (1.25, 1.25, 1.25, 1.17)
# Thumb tip -> ring knuckle (3D), in palm lengths. 97% balanced accuracy on HaGRID.
# (Tip -> pinky knuckle > 0.88, chosen from 10 photos, let 22% of fists read as Thumbs Up.)
THUMB_EXTENSION_RATIO = 0.80
# Thumb tip -> index tip in the image, in image palm lengths. Real OK signs measured a
# median of 0.35; open hands that are not OK signs stay above this for 99% of cases.
# Distance alone cannot tell a pinch from a fist, so it is only used with finger states.
PINCH_RATIO = 0.42
# A thumb within 60 degrees of vertical counts as pointing up/down.
THUMB_VERTICAL_COS = math.cos(math.radians(60))


def palm_length(points):
    """Wrist to middle-finger knuckle: the unit that makes the ratios scale-free.
    Works on 3D world points or 2D image points."""
    return math.dist(points[WRIST], points[MIDDLE_MCP])


def finger_states(world):
    """Return ``[thumb, index, middle, ring, pinky]``; True means extended."""
    palm = palm_length(world)
    if palm == 0:
        return [False] * 5
    # A folded thumb lies across the palm towards the ring finger; an extended one points away.
    thumb = math.dist(world[THUMB_TIP], world[RING_MCP]) / palm > THUMB_EXTENSION_RATIO
    fingers = [
        math.dist(world[WRIST], world[tip]) > ratio * math.dist(world[WRIST], world[pip])
        for tip, pip, ratio in zip(FINGER_TIPS, FINGER_PIPS, FINGER_EXTENSION_RATIOS)
    ]
    return [thumb] + fingers


def pinch_ratio(points_px):
    """Thumb-tip to index-tip distance in the image, in image palm lengths (only x, y are used)."""
    palm = math.dist(points_px[WRIST][:2], points_px[MIDDLE_MCP][:2])
    return math.dist(points_px[THUMB_TIP][:2], points_px[INDEX_TIP][:2]) / palm if palm else math.inf


def thumb_direction(points_px):
    """Return 'up', 'down' or 'side' for the thumb, from image points in pixels (y grows down)."""
    dx = points_px[THUMB_TIP][0] - points_px[THUMB_MCP][0]
    dy = points_px[THUMB_TIP][1] - points_px[THUMB_MCP][1]
    length = math.hypot(dx, dy)
    if length == 0:
        return "side"
    if -dy / length >= THUMB_VERTICAL_COS:
        return "up"
    if dy / length >= THUMB_VERTICAL_COS:
        return "down"
    return "side"


def classify_gesture(fingers, pinch, thumb_dir="up"):
    """
    Classify a static hand gesture.
    :param fingers: 5 booleans [Thumb, Index, Middle, Ring, Pinky] (True = extended)
    :param pinch: thumb-to-index tip distance in palm lengths (see pinch_ratio)
    :param thumb_dir: 'up', 'down' or 'side' (see thumb_direction)
    :return: gesture name
    """
    thumb, index, middle, ring, pinky = fingers
    total_fingers = sum(fingers)

    # Check proximity gesture first (OK Sign)
    if pinch < PINCH_RATIO and middle and ring and pinky:
        return "OK Sign"

    if total_fingers == 0:
        return "Fist"
    if total_fingers == 5:
        return "Open Palm"
    if fingers == [True, False, False, False, False]:
        return {"up": "Thumbs Up", "down": "Thumbs Down"}.get(thumb_dir, "Thumb Out")
    if fingers == [False, True, False, False, False]:
        return "Pointing"
    if fingers == [False, True, True, False, False]:
        return "Peace / Victory"
    if fingers == [False, True, False, False, True]:
        return "Rock On"
    if fingers == [True, True, False, False, True]:
        return "Spider-Man / ILY"
    if fingers == [True, False, False, False, True]:
        return "Call Me"
    if fingers == [True, True, False, False, False]:
        return "L Shape"
    return f"{total_fingers} Finger{'' if total_fingers == 1 else 's'} Up"


# One hand as seen by the two-hand classifier: image points in pixels, finger states,
# and that hand's single-hand gesture for this frame.
HandPose = namedtuple("HandPose", "points fingers gesture")

# Two-hand thresholds, in image palm lengths unless noted. The comments give the
# median [10th, 90th percentile] measured on HaGRID photos of that gesture.
HEART_TIP_GAP = 0.6  # thumb-thumb 0.15-0.21 [0.10, 0.38]; index-index 0.09-0.17 [0.03, 0.40]
HEART_MIN_RISE = 0.3  # index tips above thumb tips by 0.84-0.89 [0.64, 1.09]
HEART_MIN_WRIST_GAP = 1.5  # 2.6-2.8 [2.38, 2.98]; prayer hands, which also touch, measured 0.88 [0.66, 1.19]
FRAME_MAX_COS = -0.8  # index, thumb directions: -0.99, -1.00 [p90 -0.95] (exactly opposite)
FRAME_MAX_HAND_COS = -0.6  # wrist->knuckle directions: -0.96 [p90 -0.83]; hearts 0.47 / -0.01
FRAME_MAX_GAP = 3.0  # thumb-thumb 1.55 [1.32, 1.73]
TIMEOUT_MAX_TIP_TO_PALM = 1.0  # upright hand's middle tip to the flat hand's palm: 0.42 [0.23, 0.58]
CROSS_MIN_OUTWARD = 0.3  # both hands tilt away from the centre: 0.60 [0.46, 0.80]; all others <= 0.02 (p90)
CROSS_MIN_RISE = 0.3  # wrist->knuckles pointing upwards: 0.71 [0.51, 0.81]
CROSS_MAX_WRIST_GAP = 2.5  # 1.44 [1.05, 1.88]
PRAYER_MIN_HAND_COS = 0.8  # 0.98 [0.93, 1.00]
PRAYER_MAX_TIP_GAP = 0.6  # middle tips 0.11 [0.05, 0.33]
UPRIGHT_COS = 0.7  # pointing within ~45 degrees of vertical
LEVEL_COS = 0.5  # pointing within 60 degrees of horizontal


def _direction(points, a, b):
    dx, dy = points[b][0] - points[a][0], points[b][1] - points[a][1]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n else (0.0, 0.0)


def _cos(u, v):
    return u[0] * v[0] + u[1] * v[1]


def classify_two_hands(a, b):
    """Name the gesture two HandPoses make together, or None.

    Spatial gestures come first; otherwise two hands showing the same named
    single-hand gesture give "Double <gesture>" (e.g. "Double Spider-Man / ILY").
    """
    palm = (palm_length(a.points) + palm_length(b.points)) / 2
    if palm == 0:
        return None

    def gap(i):  # distance between the two hands' landmark i
        return math.dist(a.points[i][:2], b.points[i][:2]) / palm

    def facing(i, j):  # cosine between the two hands' i->j directions
        return _cos(_direction(a.points, i, j), _direction(b.points, i, j))

    up = (0.0, -1.0)  # image y grows downwards
    thumbs = a.fingers[0] and b.fingers[0]
    four_fingers = all(a.fingers[1:]) and all(b.fingers[1:])
    l_shapes = thumbs and (a.fingers[1] or b.fingers[1]) and not any(all(h.fingers[2:]) for h in (a, b))

    # Heart: thumb tips touch at the bottom, index tips touch above them, wrists apart.
    rise = (a.points[THUMB_TIP][1] + b.points[THUMB_TIP][1] - a.points[INDEX_TIP][1] - b.points[INDEX_TIP][1]) / 2 / palm
    if (thumbs and gap(THUMB_TIP) < HEART_TIP_GAP and gap(INDEX_TIP) < HEART_TIP_GAP
            and rise > HEART_MIN_RISE and gap(WRIST) > HEART_MIN_WRIST_GAP):
        return "Heart"

    # Frame: two L shapes, one rotated 180 degrees from the other, at opposite corners.
    # The index at a frame corner is often bent enough to read as curled, so one is enough;
    # open hands (middle, ring and pinky out) never form one, but otherwise look alike.
    if (l_shapes and facing(INDEX_MCP, INDEX_TIP) < FRAME_MAX_COS and facing(THUMB_MCP, THUMB_TIP) < FRAME_MAX_COS
            and facing(WRIST, MIDDLE_MCP) < FRAME_MAX_HAND_COS and gap(THUMB_TIP) < FRAME_MAX_GAP):
        return "Frame"

    # Time Out: one flat hand level, the other upright with its fingertips at the flat palm.
    if four_fingers:
        for upright, level in ((a, b), (b, a)):
            if (abs(_cos(_direction(upright.points, MIDDLE_MCP, MIDDLE_TIP), up)) > UPRIGHT_COS
                    and abs(_cos(_direction(level.points, MIDDLE_MCP, MIDDLE_TIP), up)) < LEVEL_COS):
                palm_centre = [sum(level.points[i][k] for i in PALM_POINTS) / len(PALM_POINTS) for k in (0, 1)]
                if math.dist(upright.points[MIDDLE_TIP][:2], palm_centre) / palm < TIMEOUT_MAX_TIP_TO_PALM:
                    return "Time Out"

    # X Sign (crossed arms): forearms cross, so each hand leans away from the other.
    centre_x = (a.points[WRIST][0] + b.points[WRIST][0]) / 2
    outward = min(math.copysign(1, h.points[WRIST][0] - centre_x) * _direction(h.points, WRIST, MIDDLE_MCP)[0]
                  for h in (a, b))
    rising = min(_cos(_direction(h.points, WRIST, MIDDLE_MCP), up) for h in (a, b))
    if outward > CROSS_MIN_OUTWARD and rising > CROSS_MIN_RISE and gap(WRIST) < CROSS_MAX_WRIST_GAP:
        return "X Sign"

    # Prayer: flat hands pressed together, pointing up. (Parallel hands with touching
    # fingertips also have their wrists together, so no separate wrist check is needed.)
    if (four_fingers and facing(WRIST, MIDDLE_MCP) > PRAYER_MIN_HAND_COS and rising > UPRIGHT_COS
            and gap(MIDDLE_TIP) < PRAYER_MAX_TIP_GAP):
        return "Prayer"

    if a.gesture and a.gesture == b.gesture and not a.gesture[0].isdigit():  # skip "N Fingers Up"
        return f"Double {a.gesture}"
    return None


class GestureVote:
    """Majority vote over the last ``size`` labels. The shown label only changes when
    another label strictly outvotes it, which stops single-frame flicker."""

    def __init__(self, size=5):
        self._labels = deque(maxlen=size)
        self.current = None

    def update(self, label):
        self._labels.append(label)
        counts = Counter(self._labels)
        best = max(counts.values())
        if counts[self.current] < best:
            # Among tied leaders, prefer the most recent.
            self.current = next(l for l in reversed(self._labels) if counts[l] == best)
        return self.current


class OneEuroFilter:
    """One-Euro filter (Casiez et al., CHI 2012) for a single scalar signal.

    Smooths heavily while the signal is still and lightly while it moves fast, so
    jitter is removed without making fast motion lag.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t is None:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


class LandmarkSmoother:
    """One OneEuroFilter per coordinate of every landmark."""

    def __init__(self, num_points=21, dims=3, **filter_kwargs):
        self._filters = [[OneEuroFilter(**filter_kwargs) for _ in range(dims)] for _ in range(num_points)]

    def __call__(self, points, t):
        return [tuple(f(v, t) for f, v in zip(fs, p)) for fs, p in zip(self._filters, points, strict=True)]


class HandTrack:
    """Per-hand state that has to survive across frames."""

    def __init__(self, track_id, wrist, vote_size, smoothing):
        self.id = track_id
        self.wrist = wrist
        self.missed = 0
        self.vote = GestureVote(vote_size)
        self.smoother = LandmarkSmoother(**smoothing)


class HandTracker:
    """Gives each detected hand a stable identity by matching it to the nearest wrist from
    the previous frames. This is more reliable than keying on the Left/Right label, which
    MediaPipe occasionally gives to both hands at once."""

    # Normalized-coordinate units: a hand moving fast covers ~2 frame-widths per second.
    DEFAULT_SMOOTHING = {"min_cutoff": 1.0, "beta": 10.0, "d_cutoff": 1.0}

    def __init__(self, max_jump=0.25, max_missed=10, vote_size=5, smoothing=None):
        self.max_jump = max_jump
        self.max_missed = max_missed
        self.vote_size = vote_size
        self.smoothing = smoothing or self.DEFAULT_SMOOTHING
        self.tracks = []
        self._next_id = 0

    def update(self, wrists):
        """Match this frame's wrist positions (normalized x, y) to tracks.

        Returns one HandTrack per wrist, in the same order as ``wrists``.
        """
        pairs = sorted(
            ((math.dist(w, track.wrist), i, track) for i, w in enumerate(wrists) for track in self.tracks),
            key=lambda p: (p[0], p[1], p[2].id),
        )
        assigned = [None] * len(wrists)
        used = set()
        for d, i, track in pairs:
            if d > self.max_jump:
                break
            if assigned[i] is None and track.id not in used:
                assigned[i] = track
                used.add(track.id)
        for i, w in enumerate(wrists):
            if assigned[i] is None:
                assigned[i] = HandTrack(self._next_id, w, self.vote_size, self.smoothing)
                self._next_id += 1
                self.tracks.append(assigned[i])
            assigned[i].wrist = w
            assigned[i].missed = 0
        matched = {t.id for t in assigned}
        for track in self.tracks:
            if track.id not in matched:
                track.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return assigned
