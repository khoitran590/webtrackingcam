"""Gesture classification and smoothing, independent of the camera, OpenCV and MediaPipe.

A hand is a sequence of 21 ``(x, y, z)`` points in MediaPipe's landmark order.
Finger states are computed from MediaPipe's *world* landmarks (metric 3D, centred on
the hand). They only compare 3D distances, so they do not depend on hand rotation,
palm-vs-back orientation, handedness, or distance from the camera.
"""

import math
from collections import Counter, deque

WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17
FINGER_TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky
FINGER_PIPS = (6, 10, 14, 18)

# Thresholds sit midway in the gap measured on the sample images in
# tests/test_sample_images.py, both as-is and mirrored (96 fingers, 24 thumbs).
# Wrist->tip over wrist->PIP: extended fingers measured >= 1.33, curled <= 1.16.
FINGER_EXTENSION_RATIO = 1.25
# Thumb tip -> pinky knuckle, in palm lengths: extended >= 0.98, tucked <= 0.78.
THUMB_EXTENSION_RATIO = 0.88
# Thumb tip -> index tip, in palm lengths, for the OK sign. Not calibrated on real
# OK-sign images. A fist also measured as low as 0.175, so distance alone cannot tell
# a pinch from a fist; it is only meaningful together with the finger states.
PINCH_RATIO = 0.3
# A thumb within 50 degrees of vertical counts as pointing up/down.
THUMB_VERTICAL_COS = math.cos(math.radians(50))


def palm_length(world):
    """Wrist to middle-finger knuckle: the unit that makes the ratios scale-free."""
    return math.dist(world[WRIST], world[MIDDLE_MCP])


def finger_states(world):
    """Return ``[thumb, index, middle, ring, pinky]``; True means extended."""
    palm = palm_length(world)
    if palm == 0:
        return [False] * 5
    # A folded thumb lies across the palm towards the pinky; an extended one points away.
    thumb = math.dist(world[THUMB_TIP], world[PINKY_MCP]) / palm > THUMB_EXTENSION_RATIO
    fingers = [
        math.dist(world[WRIST], world[tip]) > FINGER_EXTENSION_RATIO * math.dist(world[WRIST], world[pip])
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
    ]
    return [thumb] + fingers


def pinch_ratio(world):
    """Thumb-tip to index-tip distance in palm lengths."""
    palm = palm_length(world)
    return math.dist(world[THUMB_TIP], world[INDEX_TIP]) / palm if palm else math.inf


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
    return f"{total_fingers} Finger{'' if total_fingers == 1 else 's'} Up"


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
