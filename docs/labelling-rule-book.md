# MeshWatch — Labelling Rule Book (Phase-1, draft for geotech validation)

Last Updated: 2026-10-27

Draft rule book for annotators, and the document to validate with a geotechnical engineer. Companion to [Phase-1 Labelling Approach](phase-1-labelling-approach.md). Every definition here is a proposal. The SME's edits become the standard.

## Purpose, scope and how to use

This rule book is the single source of truth for how frames are labelled. Annotators follow it exactly; the geotechnical engineer validates and edits it. Once signed off, its definitions are the standard, and every labelling disagreement is settled by pointing back to a rule here.

Scope for this version: **Phase-1 surface segmentation** — assigning every pixel of a frame to one class (mesh, fibrecrete, rock, floor, rockfall, vent-bag, services, other, or ignore) — plus mesh-condition and corrosion flags the geotech anchors now so later phases inherit stable definitions.

How we label (full mechanics in Section 7): masks are drawn with SAM2 assistance and corrected by hand; surfaces are labelled on reduced-resolution frames (region extents do not need full resolution); fine corrosion is labelled later at full resolution.

Two things every annotator internalises before starting: (1) the **layer-precedence rule** (Section 3) decides the class where surfaces overlap, and (2) **when in doubt, mark ignore** (Section 4) rather than guess.

## Segmentation class taxonomy

Every pixel gets exactly one class. Definitions and cues below; overlaps are resolved by Section 3.

| Class | Definition | Visual cues | Common confusers (do NOT label as this) |
| --- | --- | --- | --- |
| Mesh | Steel weld or chain-link mesh visible over the surface | Regular square/diamond wire grid; wires cast shadow lines | Grid-like shotcrete cracking; wire shadows cast onto bare shotcrete |
| Fibrecrete (shotcrete) | Sprayed concrete surface coat | Smooth/hummocky grey coat, visible fibres, shrinkage cracks | Bare rock with no spray texture; mesh over shotcrete (→ mesh) |
| Rock | Exposed rock, no shotcrete or mesh | Bedding/joint texture, drill scars, no spray coat | Thinned shotcrete; muck pile (→ rockfall) |
| Floor | Intact, in-place drive floor (road base) | Compacted, rippled or wheel-tracked road base along the bottom of the frame | Loose debris or muck lying on the floor (→ rockfall) |
| Rockfall | Detached, fallen rock and floor muck | Loose angular fragments, debris heaps | In-place bedded rock (→ rock); intact floor (→ floor) |
| Vent bag | Ventilation ducting | Large smooth tube, often green, catenary sag | — |
| Services | Cables, hoses, pipes, dewatering line, signage | Linear runs along wall/back | Cable shadows on the surface behind |
| Other / background | Vehicles, people, equipment, open drive/void ahead | — | — |
| Ignore / void | Anything not reliably labelable (Section 4) | Black falloff, blur, blown glare | — |

Floor is its own class, split out of "rockfall / floor": rippled floor was the MVP's known false trigger, so the model needs to learn it explicitly.

> **TODO (SME):** decide how to label fixed structures and loose items that aren't ground support or services, such as brick vent bulkheads / stoppings, and instruments or equipment lying on the floor or against the wall. Options: `other`, `services`, a new class, or the surface behind them. In Round 0, a brick vent bulkhead was labelled `services` because no class fit.

Elements for later phases (mark now only if the SME wants them): **bolt plate**, **bolt/cable collar**. These are instances (each one separate), not surface classes — they become the Phase-2 focus.

Reference images: canonical example crops to be attached per class from the Mount Isa frames — the SME flags which examples are definitive.

## Layer precedence (which class wins on overlap)

Support stacks — mesh over shotcrete over rock, with services and vent bags in front. A pixel gets the class of the top-most thing actually visible at it. Apply this order, highest priority first:

1. **Ignore / void** — if the pixel can't be reliably read, stop here (Section 4).
2. **Services / vent bag** — foreground objects in front of the wall.
3. **Mesh** — if wire is visible at that pixel, it's mesh, whatever is behind it.
4. **Fibrecrete** — sprayed surface where no mesh wire shows.
5. **Rock** — exposed rock where there's no shotcrete or mesh.
6. **Rockfall** — detached, fallen material (debris lies on the floor, so it wins over floor).
7. **Floor** — intact, in-place drive floor.

Worked example: a shotcreted wall with mesh over it, a cable crossing, and a dark patch → cable = services, wire areas = mesh, sprayed areas between wires = fibrecrete, unreadable dark patch = ignore.

Open question for the SME (Section 8): should mesh-over-shotcrete be a single "mesh" label, or do you also need the substrate underneath recorded? For v1 we propose single-label, "mesh wins".

> **TODO (SME):** settle the mesh definition. This book says only pixels with visible wire are mesh, and the sprayed pixels between wires are fibrecrete. The labelling tool (`groundwatch-labelling`, Phase 1a) instead labels the **meshed-area extent** as one region, without cutting out the apertures, because at labelling resolution wire is about 3 px and the apertures 50–65 px, so wire-level masks can't be drawn consistently. Confirm which one is the standard, and update the hard-case rules below to match.

## The ignore / void class

Marking ignore is not failure — it keeps the labels honest and stops the model learning from pixels no one can actually read. Mark ignore when:

- The area is too dark (black falloff) or too blown-out (glare on wet rock) to identify.
- Motion blur or defocus makes the surface unreadable.
- The class is genuinely ambiguous even after applying Section 3.
- It's outside the inspection target (open drive ahead, deep void, headlight flare).

Rules: never invent a class to avoid ignore; when torn between a real class and ignore, choose ignore; ignore pixels are excluded from both training and scoring, so over-using it costs coverage but never corrupts the model. A frame that is mostly ignore should be dropped, not labelled.

> **TODO (SME):** decide whether open drive / void ahead is `other` or `ignore`. This book currently lists it under both (the Other row in Section 2, and the list above). The labelling tool treats it as `ignore` (`reason = out_of_target`), so it is excluded from training and scoring.

## Damage & condition definitions (SME-anchored)

These carry the geotechnical meaning, so they need the engineer's sign-off most. Two scales.

**Mesh condition**, mapped to the site C-MICO-TECH R-scale (damaged = R ≥ 2):

| Flag | Description | R-scale |
| --- | --- | --- |
| Sound | Installed, intact, at most minor sag | R0–R1 |
| Bagging (minor) | Localised bagging / loose panel, mesh continuous; bag < 30 cm | R2 |
| Bagging (major) / diamonding | Pronounced bagging (> 30 cm) or diamond stretching of the grid | R3 |
| Torn / broken wires | Wires broken, hanging, or missing sections | R3–R4 |
| Failed / fall-of-ground | Mesh failed with rock behind it displaced | R4–R5 |

**Corrosion (GSCI 0–4)** for plate/bolt/mesh steel — validate now, apply fully in the corrosion phase:

| Grade | Visible state | Action anchor |
| --- | --- | --- |
| 0 None | Intact coating, no oxide | Routine |
| 1 Superficial | Patchy staining, light oxide | Monitor |
| 2 Moderate | General red rust, galvanising consumed | Log, inspect next cycle |
| 3 Advanced | Layered rust, pitting, flaking at edges/washer | Prioritised physical check |
| 4 Severe | Perforation, exfoliation, corrosion + plate deformation | Rehab / replace |

**Conservative rule:** where the deciding measurement (bag depth, section loss) is not verifiable from the image, grade at the lower level and tag the instance "difficult".

## Hard-case decision rules

The recurring ambiguous situations and their ruling — this is the list to argue over with the SME:

| Situation | Ruling |
| --- | --- |
| Mesh over shotcrete | Pixels with visible wire → mesh; sprayed pixels between wires → fibrecrete (per Section 3) |
| Shotcrete thinned / patchy over rock | Sprayed coat present → fibrecrete; clearly bare rock → rock |
| Plate shotcreted over | Surface class is fibrecrete; mark the plate as an element only if its outline is discernible |
| White efflorescence / mineral staining | Not its own class — label the underlying surface; optionally note as a water indicator (see TODO below) |
| Spray-paint, survey/convergence markers, chalk | Confusers — label the surface behind; never label as damage, a plate, or a class of their own |
| Wet / specular hotspot | Surface still readable → its true class; blown out → ignore |
| Vent bag / cable across the wall | The occluder's class (vent-bag / services); do not label the wall hidden behind it |
| Rockfall vs intact floor | Loose, detached, piled → rockfall; bedded, in-place wall/back → rock; intact road base → floor |
| Frame partly blurred (drive-by) | Sharp region labelled normally; blurred region → ignore |

> **TODO (SME):** decide whether water / efflorescence should be a surface class. The perception model recommendation lists water/efflorescence among the segmentation "stuff" classes, but this book (above) and the labelling tool's schema label the surface underneath instead. Deciding to make it a class later means relabelling those regions.

## Annotation mechanics & QA

- **Tooling:** SAM2-assisted masks in CVAT; correct with polygon add/subtract and brush; export COCO-panoptic.
- **Resolution:** label surface classes on frames downscaled to \~1500–2000 px; corrosion later at full resolution.
- **Boundary tolerance:** aim for clean edges, but a few pixels of slack at a class boundary is fine — do not chase sub-pixel edges for surfaces.
- **Minimum region:** do not label specks; skip isolated regions smaller than roughly a thumbnail unless they carry a damage flag.
- **Provenance:** record per label the rater, stage (VLM pre-label / human-verified / adjudicated), confidence, and whether a VLM proposal was overridden.
- **Calibration round:** all annotators label the same 30 frames first; compare, discuss disagreements, refine this book before bulk labelling.
- **Agreement:** double-label 10–15% of frames; report pixel mIoU and, for the grades, quadratic-weighted Cohen's κ. That κ is the model's accuracy ceiling.
- **Difficult tier:** anything the annotator and SME can not confidently resolve is tagged "difficult" and set aside, never force-labelled.

## SME validation checklist & sign-off

Please review and edit directly in this doc, or leave comments. The decisions we specifically need from you:

1. **Class list** — anything missing or wrongly split (shotcrete sub-types, straps, cable bolts)?
2. **Mesh-over-shotcrete** — single "mesh" label, or record the substrate separately?
3. **Mesh R-scale mapping** — do the bagging / diamonding / torn thresholds and R-bands match site practice?
4. **GSCI rubric** — do the 0–4 descriptions and action anchors match how you'd rate a plate in the field?
5. **Efflorescence / water** — capture as its own surface class or flag, or leave for now? (TODO in Section 6)
6. **Confusers** — any site markings or fixtures we've missed that annotators would mistake for damage or support?
7. **Conservative grading** — agree with grading down when the deciding measurement isn't visible?
8. **Difficult bar** — what should escalate to you rather than be an annotator's call?
9. **Mesh definition** — visible-wire pixels, or the meshed-area extent the labelling tool uses? (TODO in Section 3)
10. **Open drive / void ahead** — `other` or `ignore`? (TODO in Section 4)
11. **Structures and loose items** — brick vent bulkheads, instruments or equipment lying around: which class? (TODO in Section 2)

