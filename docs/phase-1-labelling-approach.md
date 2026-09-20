# MeshWatch — Phase-1 Labelling Approach

Last Updated: 2026-09-19

Companion to [Perception Model Recommendation](perception-model-recommendation.md). These are working notes on how to label data for the Phase-1 surface-segmentation model (mesh / fibrecrete / rock / rockfall / vent-bag / other).

## Bottom line

The approach that seems to fit best is pixel-level masks, produced by interactive SAM-assisted region labelling — rather than bounding boxes, and rather than hand-painted pixels. An off-the-shelf, self-hosted tool (CVAT) looks like a good fit, and building a custom labeller at MVP is probably not worth it. There is also a strong case for spending the geotech engineer's time as an editor and adjudicator of model-proposed labels rather than as a pixel-pusher. The scheme decisions below — an ignore class, the mesh-over-substrate rule, a written label book — tend to matter more to final quality than either the annotation format or the tool choice.

## Annotation method: pixel-level masks, SAM-assisted

Bounding boxes look like the wrong tool for Phase 1. Mesh, fibrecrete, rock and rockfall are amorphous "stuff" that wraps the tunnel profile. An axis-aligned box around mesh on a curved wall encloses mostly non-mesh, so it gives the model almost no usable signal. Boxes fit better in Phase 2, for the countable "things" (plates, bolt collars). Semantic segmentation is per-pixel, so labels are best kept as dense masks.

"Pixel-level" does not have to mean hand-painting pixels, though. A faster route is interactive SAM2-assisted labelling: the annotator clicks a region, SAM2 returns a mask, the class is assigned and obvious errors fixed, with polygon/brush for touch-ups. That is roughly 5–20x faster than drawing from scratch. Storing raster label-maps for training while keeping the polygons for editability seems to work well.

| Method | What it gives | Cost / frame | Verdict for Phase 1 |
| --- | --- | --- | --- |
| Bounding boxes | Coarse rectangles | Low | No — cannot represent curved amorphous regions; better in Phase 2 for plates/bolts |
| Dense masks, hand-painted | Exact per-pixel labels | Very high | No — too slow, and boundaries are fuzzy anyway |
| Dense masks, SAM-assisted | Per-pixel masks, fast | Medium | Yes — the primary method here |
| Weak labels (scribbles / points / tile tags) | Partial supervision | Very low | Useful as a hybrid accelerator and for active learning, less so as the anchor set |

## Scheme decisions that matter more than the tool

This is where the geotech expertise really earns its keep.

- An explicit `ignore`/`void` class seems close to essential. The dominant image-quality problem is the black falloff and motion-blur band, and forcing a label where the surface is not legible tends to backfire; masking those pixels out of the loss (the Cityscapes "255" pattern) avoids the model learning to hallucinate confident classes into darkness.
- Mutual-exclusivity is worth deciding up front. Mesh sits over fibrecrete or over rock, so a single softmax forces a false choice. For v1, a single-label map with a clear priority rule seems to work — "label the topmost support layer" (mesh wins where wire is visible). Promoting mesh to its own binary channel (multi-label) looks worth it only if evaluation shows the downstream needs the substrate underneath.
- Labelling surfaces at reduced resolution is usually enough. Region extents do not need 24 MP precision; full-frame at ~1500–2000 px works. Full-res tile labelling is better reserved for the corrosion/fine work in Phase 2–3, where boundary precision feeds the GSCI coverage fraction.
- Sampling for diversity tends to beat runs of video. Consecutive frames are correlated, so spreading across drives and conditions, and over-sampling rare classes (rockfall), seems the better call. After v1, active learning helps — label where the model is uncertain.
- A label book and an agreement measure both pay off. One page of rules, plus reference crops per class and per hard case (mesh-over-shotcrete, efflorescence, where rockfall stops and floor starts, wet versus dry). Double-labelling 10–15% of frames and measuring pixel mIoU / kappa between annotators gives the human-agreement ceiling to judge the model against, and an honest number for the investor conversation.
- Volume. With the self-supervised backbone plus SAM assist, a usable v1 looks plausible from ~200–400 well-chosen frames rather than tens of thousands. The MVP learning curve (AUC 0.67→0.92, not plateaued) suggests more helps, but a huge set is not needed to start.

## Tooling: an off-the-shelf tool is probably enough

Building an annotation tool is a product in itself — mask editing, SAM integration, review/QA workflow, multi-user, versioning, export. That is months that do not advance the model, and a classic build-versus-buy trap at MVP.

Data governance is likely the deciding driver. Mount Isa client imagery on third-party SaaS is a real friction point, so self-hosting has the edge. The natural pick is CVAT, self-hosted: its segmentation UX and native SAM integration are a strong fit for dense masks, and the data stays on in-house infrastructure. Label Studio (also self-hostable, SAM via an ML backend) is a reasonable alternative where a more general or configurable platform is wanted, or where it is already running.

| Tool | Segmentation + SAM | Hosting / governance | Best for |
| --- | --- | --- | --- |
| CVAT (self-hosted) | Strong brush/polygon; native interactive SAM2 | Self-host — data stays in-house | Primary pick — dense masks at scale |
| Label Studio (self-hosted) | Good; SAM via ML backend, weaker mask UX | Self-host | Alternative — mixed task types, configurable UI |
| Cloud SaaS (Roboflow, Segments.ai) | Excellent SAM-assisted labelling | Cloud — data leaves premises | A quick trial only, not the corpus |

One current-state caveat is worth noting. CVAT's newest SAM3 text-prompt feature (pick a label, click one example, and it segments matching regions) is cloud-only and paid (CVAT Online / Enterprise), not self-hosted Community. Its documented failure modes — low-res, blurry boundaries, visually-similar classes, crowded scenes — line up closely with the hardest frames here. So architecting around text-prompt magic looks risky. Planning on self-hosted interactive SAM2 click-to-mask as the workhorse, and confirming current SAM2 self-host specifics for the CVAT version in use, is the safer route.

## The thin custom layer worth building

Not a tool — just glue around the off-the-shelf one, and this is where engineering time is likely best spent:

1. A pre-annotation script that runs SAM2 plus the project's VLM to generate proposal masks and imports them as pre-labels, so the geotech corrects rather than draws.
2. The domain ontology / label config, and the export-to-training-format converter.
3. The active-learning loop that pushes uncertain frames back into the labelling queue after v1.

## Spend the geotech as editor and adjudicator

This is probably the highest-leverage decision. Scarce, expensive expertise is best directed to (a) co-authoring the label book and the hard-case rules, and (b) adjudicating and correcting model/SAM proposals and inter-annotator disagreements — rather than first-pass drawing. First-pass correction of the SAM2 + VLM pre-labels can fall to the engineering team or a junior annotator; the geotech reviews, resolves ambiguous cases, and signs off.

That is verification labelling rather than de-novo labelling, and it tends to turn the geotech's hours into roughly 3–5x the labelled frames.

## Sources

- [CVAT — SAM2 available for image segmentation](https://www.cvat.ai/post/sam-model-on-cvat-cloud-now)
- [CVAT docs — Segment Anything 2 tracker / auto-annotation](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/)
- [CVAT — SAM3 text-prompt segmentation (cloud-only, paid)](https://www.cvat.ai/resources/changelog/sam3-text-prompts)
- [CVAT — best open-source data annotation tools 2026](https://www.cvat.ai/resources/blog/best-open-source-data-annotation-tools)
- [CVAT vs Labelbox vs Label Studio comparison 2026](https://welabeldata.com/blog/cvat-vs-labelbox-vs-label-studio/)

Imagery reviewed: 45 DSLR frames plus phone stills and 8K drive-by video from the Mount Isa drives (`Mining Photos and Videos/`).
