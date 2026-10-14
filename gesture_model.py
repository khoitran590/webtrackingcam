"""A small learned hand-shape classifier that works alongside the rules in gesture_logic.

The rules stay in charge of what they do well: thumb direction (up / down / out), thumb
in or out for Rock On vs. Spider-Man / ILY, finger counts for unnamed shapes, and every
two-hand gesture. The model adds what the rules cannot do: it recognises a hand that is
not making a gesture at all, and it is more accurate on shapes the rules find hard.
Its answer is used only when it is confident; otherwise the rules' label stands.

Before the model sees a hand, the hand is moved so the wrist is at the origin, scaled to
one palm length, turned so the wrist -> middle knuckle points up, and mirrored if it looks
like a left hand. A model trained without that turn recognised 98% of upright hands but
almost none turned by 90 degrees. The turn also makes thumbs up and thumbs down look the
same, which is why the model has one "thumbs" class and the rules pick the direction.

Weights live in a JSON file written by tools/train_gesture_model.py. Running the model
needs only numpy, which MediaPipe already installs.
"""

import json
import math
from pathlib import Path

import numpy as np

FEATURE_VERSION = 1
DEFAULT_CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "gesture_classifier.json"

# Model class -> label shown, for the classes that map to exactly one label.
SHAPE_LABELS = {"ok": "OK Sign", "call": "Call Me", "l_shape": "L Shape", "fist": "Fist", "palm": "Open Palm",
                "peace": "Peace / Victory", "one": "Pointing"}
THUMB_LABELS = {"up": "Thumbs Up", "down": "Thumbs Down", "side": "Thumb Out"}


def hand_features(points_px, world, image_label):
    """Position-, scale-, rotation- and handedness-free description of one hand.

    :param points_px: 21 image points in pixels (x, y[, z]); only x and y are used
    :param world: 21 MediaPipe world points (metres, axes aligned with the image)
    :param image_label: MediaPipe's handedness for the image it was given ("Left"/"Right"),
        i.e. what the hand *looks like*, not the user's hand
    """
    P = np.asarray(points_px, dtype=float)[:, :2]
    W = np.asarray(world, dtype=float)
    P = P - P[0]
    W = W - W[0]
    P /= np.linalg.norm(P[9]) or 1.0
    W /= np.linalg.norm(W[9]) or 1.0
    # Rotate in the image plane so the wrist -> middle knuckle points up (image y grows down).
    angle = math.atan2(P[9][0], -P[9][1])
    c, s = math.cos(angle), math.sin(angle)
    turn = np.array([[c, -s], [s, c]])
    P = P @ turn
    W[:, :2] = W[:, :2] @ turn
    if image_label == "Left":
        P[:, 0] *= -1
        W[:, 0] *= -1
    return np.concatenate([P.ravel(), W.ravel()])


class GestureModel:
    """Numpy forward pass of a scikit-learn MLP or logistic regression exported to JSON."""

    def __init__(self, spec):
        if spec.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"model expects features v{spec.get('feature_version')}, this code makes v{FEATURE_VERSION}")
        self.classes = list(spec["classes"])
        self.mean = np.asarray(spec["mean"], dtype=float)
        self.scale = np.asarray(spec["scale"], dtype=float)
        self.layers = [(np.asarray(layer["weights"], dtype=float), np.asarray(layer["bias"], dtype=float))
                       for layer in spec["layers"]]
        self.thresholds = {c: float(spec["thresholds"].get(c, 1.0)) for c in self.classes}
        self.info = spec.get("info", {})

    @classmethod
    def load(cls, path=DEFAULT_CLASSIFIER_PATH):
        return cls(json.loads(Path(path).read_text()))

    def probabilities(self, features):
        """Class probabilities for one feature vector (or a batch of them)."""
        x = (np.asarray(features, dtype=float) - self.mean) / self.scale
        for i, (weights, bias) in enumerate(self.layers):
            x = x @ weights + bias
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)  # ReLU between layers
        x = np.exp(x - x.max(axis=-1, keepdims=True))
        return x / x.sum(axis=-1, keepdims=True)

    def predict(self, features):
        """(class, probability) of the most likely class for one hand."""
        p = self.probabilities(features)
        best = int(np.argmax(p))
        return self.classes[best], float(p[best])


def combine(rule_label, fingers, thumb_dir, model_class, confidence, thresholds):
    """The label to show for one hand, or None when it is not making a gesture.

    The model's class is used only when its confidence reaches that class's threshold;
    otherwise the rules decide. When the model is sure the hand makes a shape the app has
    no name for ("other": a claw, a pen grip, three fingers...), a finger count from the
    rules ("3 Fingers Up") still shows, but a gesture name does not: on HaGRID the rules
    named 23% of such hands (a pen grip mostly read as Thumbs Up), this names 3%.
    """
    if confidence < thresholds.get(model_class, 1.0):
        return rule_label
    if model_class == "other":
        return rule_label if rule_label[0].isdigit() else None
    if model_class == "none":
        return None
    if model_class == "thumbs":
        return THUMB_LABELS[thumb_dir]
    if model_class == "rock":
        return "Spider-Man / ILY" if fingers[0] else "Rock On"
    return SHAPE_LABELS[model_class]
