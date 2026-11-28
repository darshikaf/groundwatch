# MeshWatch — Training Code Skeleton

Last Updated: 2026-11-26

How the training code in `src/groundwatch/` is laid out, how to run it, and the assumptions it makes where the design docs leave something open. It implements the data side of [Compute & Data Pipeline](compute-and-data-pipeline.md) and the validation scheme from [Perception Model Recommendation](perception-model-recommendation.md). The model and training loop are not built yet.

## Layout

```
configs/
  taxonomy/phase1_surface.yaml       class list: source label ids -> training indices / ignore
  experiments/phase1_surface_seg.yaml one file per experiment: data, split, tiling, augmentation
src/groundwatch/
  config.py                          loads an experiment YAML
  train.py                           `groundwatch-plan`: builds folds, reports frames / tiles / class balance
  data/
    manifest.py                      frame table (frame_id, drive_id, image, label, ...)
    splits.py                        leave-one-drive-out folds + leakage check
    taxonomy.py                      config-driven class mapping, ignore = 255
    tiling.py                        tile grid, random and label-aware crops, void-tile check, stitching
    preprocess.py                    deterministic, versioned pre-processing (calibration steps stubbed)
    augment.py                       train-time-only augmentation
    dataset.py                       FrameDataset (numpy) + as_torch_dataset adapter
    synthetic.py                     fake frames for tests and smoke runs
  models/                            empty: backbone + heads land here
scripts/make_synthetic_data.py
tests/
```

## Running it

```bash
pip install -e ".[dev]"          # add ",train" for torch once the model code lands
pytest
python scripts/make_synthetic_data.py data
groundwatch-plan configs/experiments/phase1_surface_seg.yaml --stats
```

The data layer needs only numpy, Pillow and PyYAML, so it can be developed and tested without a GPU stack.

## How it maps to the docs

| Doc requirement | Where |
| --- | --- |
| Full-res frames are the source of truth; tiles cut on the fly, never written to disk | `dataset.py` crops in `__getitem__`; LRU cache of decoded frames |
| Tile size / overlap / stride; 6000x4000 at 1024/256 = 40 tiles | `tiling.tile_grid`, `num_tiles` (tested) |
| SSL: random full-res crops, no full coverage | mode `ssl_crops` |
| Head training: crops around annotated regions, rare classes not starved | mode `train_tiles` with `LabelAwareSampler` |
| Inference: systematic overlapping grid, stitch by averaging probabilities then argmax | mode `eval_tiles` + `Stitcher` |
| Per-tile (x, y) bookkeeping | every item carries `tile`; `tile_manifest()` |
| Phase-1 surfaces may skip tiling on a ~1536 px frame | mode `downscaled` |
| Skip mostly-void tiles; void pixels out of the loss | `is_void_tile`, ignore index 255 |
| Deterministic pre-processing versioned with the model | `PREPROCESS_VERSION` |
| Grouped, leave-one-drive-out validation | `splits.leave_one_drive_out` |

## Independence from open labelling decisions

Label rasters store the labelling tool's **source ids**, and the taxonomy YAML maps them to training classes or to ignore. The open SME items in the rule book then become config edits, not code changes:

- **#5 water / efflorescence:** source id 20 is reserved and currently sent to ignore. Making it a class means adding a class entry (and relabelling, because today's labels mark the surface underneath).
- **#10 open drive ahead:** source id 21 is kept apart from `other` and currently sent to ignore, matching the labelling tool.
- **#11 bulkheads / loose equipment:** source id 22 is kept apart and currently merged into `other`.
- **#9 mesh definition** (wire pixels vs meshed-area extent) needs no code change. It changes what mesh masks look like, and so what mesh IoU means.
- Instance classes (plates, collars) and the corrosion / grading heads are Phase 2–3 and not modelled here. The `Stitcher` handles semantic classes only; the docs point to SAHI for instance merging.

## Assumptions to confirm

Each is also marked `ASSUMPTION` in the code.

1. **Source ids are placeholders.** The numbering in `phase1_surface.yaml` must be replaced by the ids the labelling export actually writes. The COCO-panoptic → raster converter is not part of this skeleton.
2. **Manifest is a CSV** with `frame_id`, `drive_id`, `image` (and optional `label`, `pass_id`, `source`, `width`, `height`). Something has to produce it from the capture store.
3. **A drive is one physical tunnel section.** Repeat passes share a `drive_id`, so a held-out drive is never seen through another pass. If the team wants pass-level grouping, split on `pass_id` instead.
4. **Validation uses a second held-out drive** when there are 3 or more drives, rotating across folds. With only a handful of drives this spends a lot of data on val; the alternative is a fixed epoch budget with no val drive (`use_val_drive: false`). How many drives the labelled set will span is the deciding number.
5. **Drives with fewer than 5 labelled frames** train but are never test drives (`min_frames_per_drive`).
6. **Labels drawn at reduced resolution** (~1500–2000 px, rule book §7) are upsampled to full-frame size with nearest-neighbour before tiling, if the aspect ratio matches within 1%.
7. **Pre-processing steps are stubs** (undistort, flat-field, colour target). They need rig calibration that does not exist yet, and it is undecided whether frames arrive already undistorted.
8. **Void-tile thresholds are placeholders** until tuned on flat-field-corrected rig imagery.
9. **Horizontal flips are on, vertical flips off** (back and floor are different classes).
10. **Class-balanced sampling** uses inverse-frequency weighting with power 0.5 and 20% plain random crops. Both are starting values to tune.
