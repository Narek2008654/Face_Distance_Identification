"""
face_utils.py — Face detection and bounding-box helpers using MediaPipe Tasks API.

Compatible with mediapipe >= 0.10 (the legacy mp.solutions API was removed in 0.10).
The required .tflite model is downloaded automatically on first use.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/"
    "blaze_face_short_range.tflite"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading MediaPipe face detection model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FaceResult:
    """Represents a single detected face."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    area: int
    confidence: float
    nose_tip: Optional[tuple[int, int]] = None  # (x, y) in pixels


# ---------------------------------------------------------------------------
# Face detector
# ---------------------------------------------------------------------------


class FaceDetector:
    """
    Thin wrapper around MediaPipe Tasks FaceDetector (mediapipe >= 0.10).

    Usage::

        with FaceDetector() as detector:
            faces = detector.detect(image_rgb)
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        self._min_confidence = min_confidence
        self._detector: Optional[mp_vision.FaceDetector] = None

    def __enter__(self) -> "FaceDetector":
        model_path = _ensure_model()
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=self._min_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        return self

    def __exit__(self, *_) -> None:
        if self._detector:
            self._detector.close()
            self._detector = None

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, image_rgb: np.ndarray) -> list[FaceResult]:
        """
        Detect faces in an HxWx3 uint8 RGB image.

        Returns a (possibly empty) list of FaceResult objects.
        """
        if self._detector is None:
            raise RuntimeError("Use FaceDetector as a context manager.")

        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self._detector.detect(mp_image)

        if not results.detections:
            return []

        faces: list[FaceResult] = []
        for det in results.detections:
            bb = det.bounding_box  # pixel coords: origin_x, origin_y, width, height
            x1 = max(0, bb.origin_x)
            y1 = max(0, bb.origin_y)
            x2 = min(w, bb.origin_x + bb.width)
            y2 = min(h, bb.origin_y + bb.height)

            area = max(0, (x2 - x1) * (y2 - y1))
            confidence = float(det.categories[0].score) if det.categories else 0.0

            # Keypoint index 2 is NOSE_TIP (normalized 0-1)
            nose_tip: Optional[tuple[int, int]] = None
            if det.keypoints and len(det.keypoints) > 2:
                kp = det.keypoints[2]
                nose_tip = (int(kp.x * w), int(kp.y * h))

            faces.append(
                FaceResult(
                    bbox=(x1, y1, x2, y2),
                    area=area,
                    confidence=confidence,
                    nose_tip=nose_tip,
                )
            )

        return faces

    # ------------------------------------------------------------------
    # Helpers (static so they can be used without an instance)
    # ------------------------------------------------------------------

    @staticmethod
    def select_largest(faces: list[FaceResult]) -> Optional[FaceResult]:
        """Return the face with the largest bounding-box area, or None."""
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)

    @staticmethod
    def shrink_bbox(
        bbox: tuple[int, int, int, int],
        fraction: float = 0.20,
        image_shape: Optional[tuple] = None,
    ) -> tuple[int, int, int, int]:
        """
        Shrink a bounding box inward by *fraction* on every side.

        Falls back to the original bbox if the shrunk box becomes degenerate.
        """
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        dx = int(bw * fraction)
        dy = int(bh * fraction)

        sx1, sy1, sx2, sy2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy

        if image_shape is not None:
            h, w = image_shape[:2]
            sx1, sx2 = max(0, sx1), min(w, sx2)
            sy1, sy2 = max(0, sy1), min(h, sy2)

        if sx2 <= sx1 or sy2 <= sy1:
            return bbox  # degenerate — return original

        return (sx1, sy1, sx2, sy2)
