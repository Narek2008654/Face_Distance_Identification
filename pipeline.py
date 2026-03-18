"""
pipeline.py — End-to-end face distance estimation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Structured output for a single detected face."""

    # Primary output
    distance_m: Optional[float]
    bbox: Optional[tuple[int, int, int, int]]

    # Metadata
    method: str          # "box" or "nose"
    num_faces: int       # total faces detected in the image
    num_valid_depth_pixels: int

    # Inner (shrunken) bbox used for depth sampling
    inner_bbox: Optional[tuple[int, int, int, int]] = None

    # Secondary estimate from nose-tip region (only when box method is used)
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
    1. Detect all faces in the image.
    2. Run Apple Depth Pro on the full image (once).
    3. For each face, sample depth inside the face region.
    4. Return one PipelineResult per face.

    Parameters
    ----------
    shrink_fraction:
        Fraction of width/height to remove from each side of the face bbox
        before sampling depth (reduces hair/background contamination).
    nose_region_fraction:
        Radius of the nose-tip sampling region as a fraction of bbox width.
    device:
        Torch device string. Auto-detected if not supplied.
    """

    def __init__(
        self,
        shrink_fraction: float = 0.20,
        nose_region_fraction: float = 0.15,
        device: Optional[str] = None,
        max_size: Optional[int] = None,
        detector_range: str = "short",
    ) -> None:
        self.shrink_fraction = shrink_fraction
        self.nose_region_fraction = nose_region_fraction
        self.max_size = max_size
        self.detector_range = detector_range
        self._depth_estimator = DepthEstimator(device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self,
        image_path: str,
        method: str = "box",
        f_px: Optional[float] = None,
    ) -> list[PipelineResult]:
        """
        Run the full pipeline and return a result for every detected face.

        Depth Pro is invoked only once regardless of how many faces are found.
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return [self._fail(method, f"Cannot read image: {image_path}")]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with FaceDetector(range_mode=self.detector_range) as detector:
            faces = detector.detect(image_rgb)

        if not faces:
            return [self._fail(method, "no face detected")]

        depth_map = self._depth_estimator.estimate_depth(
            image_rgb, f_px=f_px, max_size=self.max_size
        )

        results: list[PipelineResult] = []
        for face in faces:
            results.append(self._process_face(face, depth_map, method, len(faces), image_rgb.shape))

        return results

    def run(
        self,
        image_path: str,
        method: str = "box",
        f_px: Optional[float] = None,
    ) -> PipelineResult:
        """Run the pipeline and return the result for the largest detected face."""
        results = self.run_all(image_path, method=method, f_px=f_px)
        valid = [r for r in results if r.bbox is not None]
        if not valid:
            return results[0]
        return max(valid, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_face(
        self,
        face: FaceResult,
        depth_map: np.ndarray,
        method: str,
        num_faces: int,
        image_shape: tuple,
    ) -> PipelineResult:
        inner_bbox = FaceDetector.shrink_bbox(
            face.bbox, fraction=self.shrink_fraction, image_shape=image_shape
        )

        if method == "nose" and face.nose_tip is not None:
            sample_bbox = self._nose_bbox(face)
            used_method = "nose"
        else:
            sample_bbox = inner_bbox
            used_method = "box"

        primary = DepthEstimator.extract_region_depth(depth_map, sample_bbox)

        nose_dist: Optional[float] = None
        if used_method == "box" and face.nose_tip is not None:
            nd = DepthEstimator.extract_region_depth(depth_map, self._nose_bbox(face))
            nose_dist = nd.get("distance_m")

        return PipelineResult(
            distance_m=primary.get("distance_m"),
            bbox=face.bbox,
            method=used_method,
            num_faces=num_faces,
            num_valid_depth_pixels=primary.get("num_valid", 0),
            inner_bbox=inner_bbox,
            nose_region_distance_m=nose_dist,
            error=primary.get("error"),
        )

    def _nose_bbox(self, face: FaceResult) -> tuple[int, int, int, int]:
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
