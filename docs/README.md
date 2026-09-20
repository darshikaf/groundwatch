# GroundWatch Design Docs

Working notes for the MeshWatch perception system: detecting and grading ground-support damage (mesh, plates, bolts, corrosion) in underground mining drives from camera imagery. These are drafts for discussion and validation, not final specifications.

## Find your doc by role

**Geotechnical engineer (subject-matter expert)** — the definitions that carry geotechnical meaning need validation:

- [Labelling Rule Book](labelling-rule-book.md) — the class taxonomy, the mesh-condition (R-scale) and corrosion (GSCI 0–4) grade definitions, and the hard-case rulings to be reviewed. The sign-off checklist at the end lists the specific decisions needed.
- [Phase-1 Labelling Approach](phase-1-labelling-approach.md) — the reasoning behind how data is labelled, and where geotech time is best spent.

**Camera rig** — what the imagery needs to support the model:

- [Camera Requirements](camera-requirements.md) — the camera choices that affect model accuracy (resolution, shutter, colour, lighting, stereo), with a suggested order for cutting scope under budget.

**Computer-vision / ML team** — the model and the infrastructure around it:

- [Perception Model Recommendation](perception-model-recommendation.md) — the model architecture: a multi-task stack on a shared self-supervised backbone, and what to build in which phase.
- [Compute & Data Pipeline](compute-and-data-pipeline.md) — capture resolution, pre-processing, tiling, and the GPU and storage cost of training (with indicative on-demand GPU prices).
- [Phase-1 Labelling Approach](phase-1-labelling-approach.md) — annotation method and tooling for the first model.

## All documents

| Document | Covers | Primary audience |
| --- | --- | --- |
| [Camera Requirements](camera-requirements.md) | Camera and illumination targets, and why each matters to the model | Camera rig |
| [Compute & Data Pipeline](compute-and-data-pipeline.md) | Resolution, pre-processing, tiling, GPU and storage sizing | CV / ML |
| [Perception Model Recommendation](perception-model-recommendation.md) | Model architecture, task breakdown, phased delivery | CV / ML |
| [Phase-1 Labelling Approach](phase-1-labelling-approach.md) | How to label Phase-1 data; annotation method and tooling | CV / ML, geotech |
| [Labelling Rule Book](labelling-rule-book.md) | Class definitions, grade scales, hard-case rulings, sign-off | Geotech, annotators |
