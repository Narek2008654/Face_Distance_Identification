"""
pipeline.py — End-to-end face distance estimation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from depth_utils import DepthEstimator
from face_utils import FaceDetector, FaceResult


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Structured output from one pipeline run."""

    # Primary output
    distance_m: Optional[float]
    bbox: Optional[tuple[int, int, int, int]]

    # Metadata
    method: str                    # "box" or "nose"
    num_faces: int
    num_valid_depth_pixels: int

    # Inner (shrunken) bbox used for depth sampling
    inner_bbox: Optional[tuple[int, int, int, int]] = None

    # Secondary estimate from nose-tip region (always computed when box method is used)
    nose_region_distance_m: Optional[float] = None

    # Populated on failure
    error: Optional[str] = None

    def ok(self) -> bool:
        return self.distance_m is not None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class FaceDistancePipeline:
    """
    Monocular face distance estimation.

    Steps
    -----
    1. Detect faces in the image using MediaPipe.
    2. Select the largest face.
    3. Run Apple Depth Pro on the full image.
    4. Sample depth inside the selected face region.
    5. Return a single scalar distance in metres.

    Parameters
    ----------
    shrink_fraction:
        Fraction of width/height to remove from each side of the face bbox
        before sampling depth (reduces hair/background contamination).
    nose_region_fraction:
        Radius of the nose-tip sampling region expressed as a fraction of
        the face bounding-box width.
    device:
        Torch device string.  Auto-detected if not supplied.
    """

    def __init__(
        self,
        shrink_fraction: float = 0.20,
        nose_region_fraction: float = 0.15,
        device: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.shrink_fraction = shrink_fraction
        self.nose_region_fraction = nose_region_fraction
        self.max_size = max_size
        self._depth_estimator = DepthEstimator(device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image_path: str,
        method: str = "box",
        f_px: Optional[float] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline on a single image file.

        Parameters
        ----------
        image_path:
            Path to the input RGB image.
        method:
            ``"box"``  — use the shrunken face bounding box (default).
            ``"nose"`` — use a small region around the nose-tip keypoint.
        f_px:
            Optional camera focal length in pixels for Depth Pro.

        Returns
        -------
        PipelineResult
        """
        # --- Load image -------------------------------------------------------
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return self._fail(
                method, f"Cannot read image: {image_path}"
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # --- Face detection ---------------------------------------------------
        with FaceDetector() as detector:
            faces = detector.detect(image_rgb)

        if not faces:
            return self._fail(method, "no face detected")

        face: FaceResult = FaceDetector.select_largest(faces)

        inner_bbox = FaceDetector.shrink_bbox(
            face.bbox,
            fraction=self.shrink_fraction,
            image_shape=image_rgb.shape,
        )

        # --- Depth estimation -------------------------------------------------
        depth_map = self._depth_estimator.estimate_depth(image_rgb, f_px=f_px, max_size=self.max_size)

        # --- Choose primary sampling region -----------------------------------
        if method == "nose" and face.nose_tip is not None:
            sample_bbox = self._nose_bbox(face)
        else:
            sample_bbox = inner_bbox
            method = "box"  # normalise: fall back to box if nose unavailable

        primary = DepthEstimator.extract_region_depth(depth_map, sample_bbox)

        # --- Secondary nose estimate (informational, only when using box) -----
        nose_dist: Optional[float] = None
        if method == "box" and face.nose_tip is not None:
            nose_bbox = self._nose_bbox(face)
            nd = DepthEstimator.extract_region_depth(depth_map, nose_bbox)
            nose_dist = nd.get("distance_m")

        return PipelineResult(
            distance_m=primary.get("distance_m"),
            bbox=face.bbox,
            method=method,
            num_faces=len(faces),
            num_valid_depth_pixels=primary.get("num_valid", 0),
            inner_bbox=inner_bbox,
            nose_region_distance_m=nose_dist,
            error=primary.get("error"),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nose_bbox(self, face: FaceResult) -> tuple[int, int, int, int]:
        """Build a small square bbox centred on the nose tip."""
        nx, ny = face.nose_tip
        bw = face.bbox[2] - face.bbox[0]
        r = max(5, int(bw * self.nose_region_fraction))
        return (nx - r, ny - r, nx + r, ny + r)

    @staticmethod
    def _fail(method: str, error: str) -> PipelineResult:
        return PipelineResult(
            distance_m=None,
            bbox=None,
            method=method,
            num_faces=0,
            num_valid_depth_pixels=0,
            error=error,
        )
