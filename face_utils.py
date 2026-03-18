"""
face_utils.py — Face detection and bounding-box helpers.

Two detector backends:
  "short"  — MediaPipe BlazeFace short-range (mediapipe >= 0.10).
             Best for close-up / selfie images.
  "full"   — OpenCV Haar cascade (bundled with cv2, no download needed).
             Better for group photos and distant faces.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# ---------------------------------------------------------------------------
# Model download (short-range MediaPipe model only)
# ---------------------------------------------------------------------------

_SHORT_RANGE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/"
    "blaze_face_short_range.tflite"
)
_SHORT_RANGE_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")


def _ensure_mediapipe_model() -> str:
    if not os.path.exists(_SHORT_RANGE_PATH):
        print(f"Downloading MediaPipe face detection model to {_SHORT_RANGE_PATH} ...")
        urllib.request.urlretrieve(_SHORT_RANGE_URL, _SHORT_RANGE_PATH)
        print("Download complete.")
    return _SHORT_RANGE_PATH


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
    Face detector with two backends selectable via ``range_mode``.

    range_mode="short" (default)
        MediaPipe BlazeFace short-range — fast, accurate for close-up faces.
    range_mode="full"
        OpenCV Haar cascade — handles group photos and small/distant faces.
        No extra downloads needed; bundled with OpenCV.

    Usage::

        with FaceDetector(range_mode="full") as detector:
            faces = detector.detect(image_rgb)
    """

    def __init__(self, min_confidence: float = 0.5, range_mode: str = "short") -> None:
        self._min_confidence = min_confidence
        self._range_mode = range_mode
        self._detector: Optional[mp_vision.FaceDetector] = None
        self._cascade: Optional[cv2.CascadeClassifier] = None

    def __enter__(self) -> "FaceDetector":
        if self._range_mode == "full":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
        else:
            model_path = _ensure_mediapipe_model()
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
        self._cascade = None

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, image_rgb: np.ndarray) -> list[FaceResult]:
        """
        Detect faces in an HxWx3 uint8 RGB image.

        Returns a (possibly empty) list of FaceResult objects.
        """
        if self._range_mode == "full":
            return self._detect_haar(image_rgb)
        return self._detect_mediapipe(image_rgb)

    def _detect_mediapipe(self, image_rgb: np.ndarray) -> list[FaceResult]:
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

            faces.append(FaceResult(bbox=(x1, y1, x2, y2), area=area,
                                    confidence=confidence, nose_tip=nose_tip))

        return faces

    def _detect_haar(self, image_rgb: np.ndarray) -> list[FaceResult]:
        if self._cascade is None:
            raise RuntimeError("Use FaceDetector as a context manager.")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
        )

        faces: list[FaceResult] = []
        if len(detections) == 0:
            return faces

        for (x, y, fw, fh) in detections:
            x1, y1, x2, y2 = int(x), int(y), int(x + fw), int(y + fh)
            area = fw * fh
            faces.append(FaceResult(bbox=(x1, y1, x2, y2), area=area, confidence=1.0))

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
