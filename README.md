# groundwatch

Automated detection and grading of ground-support damage in underground mining tunnels
from vehicle-mounted camera imagery — replacing the manual, walk-the-drive inspection
currently done by geotechnicians.

Working name for the perception system: **MeshWatch**. A feasibility MVP has shown that
damaged support mesh can be detected automatically from tunnel imagery (patch classifier,
AUC 0.83 leave-one-clip-out, learning curve still climbing). This repository is the home
for the production computer-vision work that follows: the model training code, and the
design docs that specify it.

## What the system does

A rig on a mine vehicle captures the tunnel walls and back; computer-vision models detect
and grade mesh damage (R0–R5) and ground-support corrosion (GSCI 0–4), then map the results
onto the tunnel for inspectors to review remotely. It keeps people out of high-risk zones,
gives consistent repeatable grading instead of by-eye judgement, and builds a deterioration
timeline across repeat passes.

## Repository layout

- `docs/` — design documents for the perception stack (see below).
- `src/groundwatch/` — training code: manifest, leave-one-drive-out splits, tiling and stitching for 24 MP frames, datasets. Model code not yet built. See [`docs/training-code.md`](docs/training-code.md).
- `configs/` — class taxonomy and experiment configs.
- `tests/` — `pytest`.

## Documents

- [`docs/perception-model-recommendation.md`](docs/perception-model-recommendation.md) — the model architecture: a multi-task perception stack (surface segmentation, instance detection, corrosion grading) on one shared self-supervised backbone, delivered in phases.
- [`docs/phase-1-labelling-approach.md`](docs/phase-1-labelling-approach.md) — how to label Phase-1 data: pixel-level masks via SAM-assisted annotation in CVAT, with the geotech as editor/adjudicator.
- [`docs/camera-requirements.md`](docs/camera-requirements.md) — plain-language input to the rig team on what the cameras must deliver (resolution, global shutter, colour, lighting, stereo) and why.
- [`docs/compute-and-data-pipeline.md`](docs/compute-and-data-pipeline.md) — capture resolution, pre-processing, tiling, and the GPU/storage budget for training and inference.
- [`docs/training-code.md`](docs/training-code.md) — layout of the training code, how to run it, and the assumptions it makes.

## Status

Pre-training / MVP-proven. Next: lock the label taxonomy, stand up the SAM2 + VLM labelling
pipeline, continue-pretrain the backbone, and train and validate Phase-1 surface segmentation
with leave-one-drive-out splits.
