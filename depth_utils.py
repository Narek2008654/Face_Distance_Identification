"""
depth_utils.py — Depth Pro integration for metric depth estimation.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Depth estimator
# ---------------------------------------------------------------------------


class DepthEstimator:
    """
    Wraps Apple ml-depth-pro for single-image metric depth estimation.

    The model is loaded lazily on the first call to ``estimate_depth``.
    Re-using the same instance for multiple images avoids repeated model loads.

    Parameters
    ----------
    device:
        Torch device string (``"cuda"``, ``"cpu"``, ``"mps"``).
        Auto-detected from CUDA availability if not supplied.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._model = None
        self._transform = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load Depth Pro weights into memory (called automatically on first use)."""
        try:
            import depth_pro
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig
        except ImportError as exc:
            raise ImportError(
                "depth_pro is not installed.\n"
                "Follow the README instructions:\n"
                "  git clone https://github.com/apple/ml-depth-pro\n"
                "  cd ml-depth-pro && pip install -e .\n"
                "  # Download model weights per the Depth Pro README"
            ) from exc

        # The default config uses a relative path ("./checkpoints/depth_pro.pt")
        # which breaks when running from outside the ml-depth-pro directory.
        # Resolve to an absolute path based on the installed package location.
        import importlib.util, dataclasses
        pkg_file = importlib.util.find_spec("depth_pro").origin  # …/src/depth_pro/__init__.py
        pkg_dir = os.path.dirname(pkg_file)                      # …/src/depth_pro/
        repo_dir = os.path.dirname(os.path.dirname(pkg_dir))     # …/ml-depth-pro/
        checkpoint = os.path.join(repo_dir, "checkpoints", "depth_pro.pt")

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"Depth Pro checkpoint not found at: {checkpoint}\n"
                "Download it by running get_pretrained_models.sh inside the ml-depth-pro repo."
            )

        # Verify the file is a valid zip archive (PyTorch checkpoints are zip files).
        # A corrupted/partial download would otherwise produce a cryptic RuntimeError deep in torch.load.
        import zipfile
        if not zipfile.is_zipfile(checkpoint):
            size_mb = os.path.getsize(checkpoint) / 1024 ** 2
            raise RuntimeError(
                f"Depth Pro checkpoint is corrupted or incomplete ({size_mb:.0f} MB).\n"
                f"Expected ~1800 MB. Delete it and re-download:\n"
                f"  rm {checkpoint}\n"
                f"  # then re-run the script or get_pretrained_models.sh"
            )

        config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=checkpoint)
        self._model, self._transform = depth_pro.create_model_and_transforms(
            config=config, device=self.device
        )
        self._model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def estimate_depth(
        self,
        image_rgb: np.ndarray,
        f_px: Optional[float] = None,
        max_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run Depth Pro on an RGB image and return a metric depth map.

        Parameters
        ----------
        image_rgb:
            HxWx3 uint8 numpy array in RGB order.
        f_px:
            Camera focal length in pixels.  Pass ``None`` to let Depth Pro
            estimate it internally (the default and recommended mode).
        max_size:
            If set, the image is downscaled so its longest side is at most
            this many pixels before being passed to Depth Pro, then the
            resulting depth map is upscaled back to the original resolution.
            Useful for CPU inference where full-resolution images are very slow
            (e.g. ``--max-size 512`` gives a large speed-up at some quality cost).

        Returns
        -------
        depth_map:
            Float32 numpy array of shape (H, W) with depth values in metres.
            Invalid/undefined pixels are represented as 0 or NaN.
        """
        if self._model is None:
            self.load_model()

        h, w = image_rgb.shape[:2]

        # Optionally downscale for faster CPU inference
        if max_size is not None and max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            infer_image = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Adjust focal length proportionally if provided
            infer_f_px = f_px * scale if f_px is not None else None
        else:
            infer_image = image_rgb
            infer_f_px = f_px

        pil_image = Image.fromarray(infer_image)
        input_tensor = self._transform(pil_image)

        with torch.no_grad():
            prediction = self._model.infer(input_tensor, f_px=infer_f_px)

        depth: torch.Tensor = prediction["depth"]
        depth_np: np.ndarray = depth.squeeze().cpu().float().numpy()

        # Always resize depth map back to the original image resolution
        if depth_np.shape != (h, w):
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth_np

    # ------------------------------------------------------------------
    # Region extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_region_depth(
        depth_map: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> dict:
        """
        Extract a robust scalar depth estimate from a rectangular region.

        Uses the **median** of all valid (positive, finite) depth pixels.

        Parameters
        ----------
        depth_map:
            Full-image depth map from ``estimate_depth``.
        bbox:
            ``(x1, y1, x2, y2)`` in pixels, origin at top-left.

        Returns
        -------
        dict with keys:
            - ``distance_m`` (float | None) — median depth in metres
            - ``num_valid``  (int)           — number of valid pixels used
            - ``error``      (str | None)    — reason string if failed
        """
        x1, y1, x2, y2 = bbox
        H, W = depth_map.shape

        # Clamp to array bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            return {"distance_m": None, "num_valid": 0, "error": "empty region after clamping"}

        region = depth_map[y1:y2, x1:x2]
        valid_mask = (region > 0) & np.isfinite(region)
        valid_values = region[valid_mask]

        if valid_values.size == 0:
            return {"distance_m": None, "num_valid": 0, "error": "no valid depth values in region"}

        return {
            "distance_m": float(np.median(valid_values)),
            "num_valid": int(valid_values.size),
            "error": None,
        }
