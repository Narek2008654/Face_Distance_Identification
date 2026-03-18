# Face Distance Estimator

Estimates the metric distance (in metres) from the camera to the primary human face in a single RGB image.

**Pipeline:**

1. Detect faces → MediaPipe Face Detector (pretrained)
2. Select the largest face
3. Run Apple Depth Pro on the full image → dense metric depth map
4. Sample the median depth inside the (shrunken) face region
5. Return distance in metres + bounding box

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU is strongly recommended (Depth Pro is a large ViT model). CPU inference works but is slow.

---

## Setup

### 1 — Install Apple ml-depth-pro

Depth Pro is not on PyPI. Clone and install it from source:

```bash
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .
```

Then download the pretrained weights using the script provided in that repo:

```bash
# From inside the ml-depth-pro directory:
source get_pretrained_models.sh
```

This downloads `checkpoints/depth_pro.pt` (~1.3 GB).

> The checkpoint path is baked into the Depth Pro package's default config.
> Keep the weights at `ml-depth-pro/checkpoints/depth_pro.pt` (relative to the
> installed package) so that `depth_pro.create_model_and_transforms()` can find them.

### 2 — Install remaining dependencies

From the root of *this* project:

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic

```bash
python main.py --image person.jpg
```

### With visualisation output

```bash
python main.py --image person.jpg --output result.jpg
```

### Nose-tip sampling instead of face-box sampling

```bash
python main.py --image person.jpg --method nose
```

### Provide camera focal length (optional)

```bash
python main.py --image person.jpg --focal-length 1200
```

If `--focal-length` is omitted, Depth Pro estimates it automatically.

### Full options

```
usage: face_distance [-h] --image PATH [--method {box,nose}]
                     [--focal-length F_PX] [--output PATH]
                     [--device DEVICE] [--shrink FRAC]

options:
  --image PATH          Input image file (JPEG, PNG, …)
  --method {box,nose}   Depth sampling strategy (default: box)
  --focal-length F_PX   Camera focal length in pixels
  --output PATH         Save annotated image to this path
  --device DEVICE       PyTorch device: cuda / cpu / mps
  --shrink FRAC         Face bbox inward shrink fraction (default: 0.20)
```

---

## Example output

```
----------------------------------------
  Faces detected       : 1
  Face bbox (px)       : x1=312 y1=148 x2=554 y2=442
  Sampling bbox (px)   : x1=360 y1=196 x2=506 y2=394
  Method               : box
  Distance             : 1.843 m
  Valid depth pixels   : 14352
  Nose-region estimate : 1.791 m
----------------------------------------
Saved visualisation → result.jpg
```

The annotated image contains:
- **Green rectangle** — detected face bounding box
- **Orange rectangle** — shrunken region used for depth sampling
- **Distance label** — overlaid at the top of the face box

---

## Project structure

```
.
├── main.py          # CLI entry point
├── pipeline.py      # Orchestrates detection + depth → distance
├── face_utils.py    # MediaPipe face detection + bbox helpers
├── depth_utils.py   # Depth Pro wrapper + region depth extraction
├── requirements.txt
└── README.md
```

---

## How it works

### Face detection
MediaPipe's `FaceDetection` model (model_selection=1, range up to ~5 m) returns
axis-aligned bounding boxes and 6 keypoints including the nose tip.

### Bounding-box shrinkage
The raw MediaPipe box often includes hair, ears, and background.
The pipeline shrinks each side inward by `--shrink` (default 20 %) before
sampling depth, reducing noise at the edges.

### Depth estimation
Apple [Depth Pro](https://github.com/apple/ml-depth-pro) is a zero-shot
metric monocular depth model.  It produces a dense (H×W) depth map in metres
without requiring camera calibration — it estimates the focal length internally.

### Distance scalar
The **median** of all valid (positive, finite) depth pixels inside the selected
region is used as the final distance.  The median is robust to outliers caused
by specular highlights, hair strands crossing the box boundary, or occasional
Depth Pro artefacts.

---

## Extending the project

| Goal | Where to change |
|---|---|
| Replace face detector | Subclass or rewrite `face_utils.FaceDetector` |
| Add webcam support | Call `pipeline.run()` per frame with a temp-file or adapt `depth_utils` to accept `np.ndarray` paths |
| Batch processing | Loop over images and collect `PipelineResult` objects |
| Track multiple faces | Remove `select_largest`; iterate over all `FaceResult` objects |

---

## Known limitations

- Depth Pro is a large model (~1.3 GB weights).  First-run load time is several seconds.
- Very small faces (<50 px wide) may produce unreliable depth estimates.
- Accuracy degrades on heavily occluded faces or faces at extreme angles.
- The model's metric scale is calibrated for typical photographic scenes;
  accuracy may vary in unusual imaging conditions.
