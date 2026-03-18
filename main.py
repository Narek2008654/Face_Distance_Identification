"""
main.py — CLI entry point for monocular face distance estimation.

Example
-------
    python main.py --image person.jpg
    python main.py --image person.jpg --method nose --output result.jpg
    python main.py --image person.jpg --focal-length 1200 --debug
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from face_utils import FaceDetector
from pipeline import FaceDistancePipeline, PipelineResult


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def draw_result(image_path: str, result: PipelineResult, output_path: str) -> None:
    """Annotate the image with the face box and distance, then save it."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[warn] Cannot reload image for visualisation: {image_path}", file=sys.stderr)
        return

    green = (0, 220, 0)
    orange = (0, 165, 255)  # BGR
    white = (255, 255, 255)

    if result.bbox:
        x1, y1, x2, y2 = result.bbox
        # Outer face bbox — green
        cv2.rectangle(image, (x1, y1), (x2, y2), green, 2)

    if result.inner_bbox:
        ix1, iy1, ix2, iy2 = result.inner_bbox
        # Inner (depth-sampling) bbox — orange dashed approximation
        cv2.rectangle(image, (ix1, iy1), (ix2, iy2), orange, 1)

    if result.distance_m is not None and result.bbox:
        label = f"{result.distance_m:.2f} m  [{result.method}]"
        x1, y1 = result.bbox[:2]
        text_y = max(y1 - 12, 20)
        # Shadow for readability
        cv2.putText(image, label, (x1 + 1, text_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, green, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, image)
    print(f"Saved visualisation → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face_distance",
        description="Estimate camera-to-face distance from a single RGB image "
                    "using MediaPipe face detection and Apple Depth Pro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, metavar="PATH",
                   help="Input image file (JPEG, PNG, …)")
    p.add_argument("--method", choices=["box", "nose"], default="box",
                   help="Depth sampling strategy: "
                        "'box' = shrunken face bbox, "
                        "'nose' = small region around nose-tip keypoint")
    p.add_argument("--focal-length", type=float, default=None, metavar="F_PX",
                   help="Camera focal length in pixels.  "
                        "If omitted Depth Pro estimates it automatically.")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Save annotated image to this path (e.g. out.jpg). "
                        "Skipped if not specified.")
    p.add_argument("--device", default=None, metavar="DEVICE",
                   help="PyTorch device: 'cuda', 'cpu', or 'mps'.  "
                        "Auto-detected if not set.")
    p.add_argument("--shrink", type=float, default=0.20, metavar="FRAC",
                   help="Fraction of face bbox to shrink inward on each side.")
    p.add_argument("--max-size", type=int, default=None, metavar="PX",
                   help="Downscale the image so its longest side is at most PX pixels "
                        "before running Depth Pro, then upscale the depth map back. "
                        "Recommended for CPU-only inference (e.g. 512 or 768).")
    return p


def print_result(result: PipelineResult) -> None:
    sep = "-" * 40
    print(sep)
    print(f"  Faces detected       : {result.num_faces}")
    if result.bbox:
        x1, y1, x2, y2 = result.bbox
        print(f"  Face bbox (px)       : x1={x1} y1={y1} x2={x2} y2={y2}")
    if result.inner_bbox:
        ix1, iy1, ix2, iy2 = result.inner_bbox
        print(f"  Sampling bbox (px)   : x1={ix1} y1={iy1} x2={ix2} y2={iy2}")
    print(f"  Method               : {result.method}")
    if result.distance_m is not None:
        print(f"  Distance             : {result.distance_m:.3f} m")
    else:
        print(f"  Distance             : N/A  ({result.error})")
    print(f"  Valid depth pixels   : {result.num_valid_depth_pixels}")
    if result.nose_region_distance_m is not None:
        print(f"  Nose-region estimate : {result.nose_region_distance_m:.3f} m")
    print(sep)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = FaceDistancePipeline(
        shrink_fraction=args.shrink,
        device=args.device,
        max_size=args.max_size,
    )

    result = pipeline.run(
        image_path=args.image,
        method=args.method,
        f_px=args.focal_length,
    )

    print_result(result)

    if args.output:
        draw_result(args.image, result, args.output)

    if not result.ok():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
