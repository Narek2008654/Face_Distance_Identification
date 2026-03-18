"""
main.py — CLI entry point for monocular face distance estimation.

Example
-------
    python main.py --image person.jpg
    python main.py --image people.jpg --detector-model full --output result.jpg
    python main.py --image person.jpg --method nose --focal-length 1200
"""

from __future__ import annotations

import argparse
import sys

import cv2

from pipeline import FaceDistancePipeline, PipelineResult


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def draw_results(image_path: str, results: list[PipelineResult], output_path: str) -> None:
    """Annotate the image with all face boxes and distances, then save it."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[warn] Cannot reload image for visualisation: {image_path}", file=sys.stderr)
        return

    green = (0, 220, 0)
    orange = (0, 165, 255)  # BGR

    for result in results:
        if result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), green, 2)

        if result.inner_bbox:
            ix1, iy1, ix2, iy2 = result.inner_bbox
            cv2.rectangle(image, (ix1, iy1), (ix2, iy2), orange, 1)

        if result.distance_m is not None and result.bbox:
            label = f"{result.distance_m:.2f} m"
            x1, y1 = result.bbox[:2]
            text_y = max(y1 - 8, 16)
            cv2.putText(image, label, (x1 + 1, text_y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, image)
    print(f"Saved visualisation → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face_distance",
        description="Estimate camera-to-face distance from a single RGB image "
                    "using face detection and Apple Depth Pro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, metavar="PATH",
                   help="Input image file (JPEG, PNG, …)")
    p.add_argument("--method", choices=["box", "nose"], default="box",
                   help="Depth sampling strategy: "
                        "'box' = shrunken face bbox, "
                        "'nose' = small region around nose-tip keypoint")
    p.add_argument("--focal-length", type=float, default=None, metavar="F_PX",
                   help="Camera focal length in pixels. "
                        "If omitted Depth Pro estimates it automatically.")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Save annotated image to this path (e.g. out.jpg). "
                        "Skipped if not specified.")
    p.add_argument("--device", default=None, metavar="DEVICE",
                   help="PyTorch device: 'cuda', 'cpu', or 'mps'. "
                        "Auto-detected if not set.")
    p.add_argument("--shrink", type=float, default=0.20, metavar="FRAC",
                   help="Fraction of face bbox to shrink inward on each side.")
    p.add_argument("--detector-model", choices=["short", "full"], default="short",
                   metavar="RANGE",
                   help="Face detector: 'short' (close-up, default) "
                        "or 'full' (group photos / distant faces).")
    p.add_argument("--max-size", type=int, default=None, metavar="PX",
                   help="Downscale the image so its longest side is at most PX pixels "
                        "before running Depth Pro, then upscale the depth map back. "
                        "Recommended for CPU-only inference (e.g. 512 or 768).")
    return p


def nearest(results: list[PipelineResult]) -> PipelineResult:
    """Return the face closest to the camera."""
    valid = [r for r in results if r.distance_m is not None]
    if not valid:
        return results[0]
    return min(valid, key=lambda r: r.distance_m)


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
        detector_range=args.detector_model,
    )

    results = pipeline.run_all(
        image_path=args.image,
        method=args.method,
        f_px=args.focal_length,
    )

    result = nearest(results)
    print_result(result)

    if args.output:
        draw_results(args.image, [result], args.output)

    return 0 if result.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
