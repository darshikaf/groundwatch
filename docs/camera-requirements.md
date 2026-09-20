# MeshWatch — Camera Requirements (Data-Science Input to the Rig Team)

Last Updated: 2026-09-20

## Purpose and framing

This is the computer-vision (CV) input to the rig design. Sensor and optics selection, exposure and sync, mechanical integration, illumination, and localisation are TBD. This document covers only the CV-specific part: how each camera choice affects model accuracy, and which choices matter most for detection and corrosion grading.

The core idea: the camera sets a ceiling the models cannot raise. More data, better architectures, and self-supervised pretraining all help, but none of them recover signal the sensor never captured or a codec threw away. A few capture choices are effectively irreversible. This document flags those, with reasoning, and puts numbers on the targets.

Where the pixels go, for context. Each frame is colour-corrected against an in-frame reference card, then cut into overlapping full-resolution tiles (a 20+ MP frame will not fit a network whole). The tiles run through detection and segmentation to find mesh, plates, and bolts. Each element then gets a per-pixel corrosion read, combined into a single grade (GSCI 0–4). Grading is the most demanding stage, so it drives most of the requirements below.

Scope for this stage is cameras and illumination only. No LiDAR, IMU, odometry, or survey control.

## Target vs floor

Read the requirements as a north star, not a minimum. The table describes the full-fidelity production system; the MVP can sit below it. Two kinds of requirement differ in nature:

- **Hard floor:** global shutter, controlled lighting and blur, stable colour plus the in-frame reference card, and raw / ≥10-bit format. These protect the signal.
- **Tier-down targets:** detail level (~0.5 mm/px), lens sharpness, and stereo. These scale with ambition. A coarser, single-camera MVP mainly limits how finely corrosion can be graded and drops distance measurement — both recoverable with a better rig later.


## Camera requirements

| Parameter | Target | CV rationale |
| --- | --- | --- |
| Detail level (resolution at the wall) | ~0.5 mm per pixel (0.3–0.4 is better) | Grading corrosion is the hardest task. Distinction between light rust and pitting needs 1–3 mm features to show up as several pixels. Too few pixels and the grades blur together, which no model can recover. Finding mesh and plates is far easier and needs less detail. Specify this in mm-per-pixel, not megapixels. |
| Lens sharpness | A lens sharp enough to match the sensor at that detail level | A soft lens wastes a high-resolution sensor. Sharpness at the fine-rust scale matters more than the megapixel count. |
| Shutter | Global shutter, synced across both cameras | A rolling shutter records the image row by row, which smears texture on a moving rig and breaks stereo. Treat as essential. |
| Motion blur | Under 1 pixel of blur at speed | Blur wipes out the fine rust texture. Freeze motion with a short light flash rather than a long exposure. Rough guide: at 8.3 m/s, staying under half a pixel needs about a 30 µs flash. |
| Colour | Colour sensor, steady lighting and white balance, and a small colour reference card in every frame | Rust is judged largely by colour. If colour drifts between drives, the model learns the lighting instead of the rust. The card lets every image be corrected to the same colours. |
| File format | Raw or high-quality images (10–12-bit) for graded frames; avoid heavily compressed video for anything graded | Compression throws away fine detail and adds artifacts the model can latch onto. Compressed video is fine for the coarser detection tasks. |
| Dynamic range | A high-dynamic-range sensor; expose to keep bright wet surfaces from blowing out | One light source makes very bright and very dark areas in the same frame. Blown-out or crushed areas cannot be graded and get discarded. More usable range means more gradeable area per frame, and fewer frames to hand-label. |
| Stereo cameras | A synced pair, spaced for the 1.5–3 m working distance | One camera cannot measure real distances such as bolt spacing or mesh sag. One camera is an acceptable budget start, but loses the measurements. |
| Distance and overlap | A steady 1.5–3 m from the wall, with overlap between shots | Steady distance keeps the detail level constant, which the models rely on. Overlap lets shots be stitched together and the same bolt be tracked across passes. |

Optional but helpful: cross-polarising filters on the lights and lenses. Wet rock throws bright glare that hides the surface; the filters cut the glare and recover the texture underneath, at a cost of roughly 1.5–2 stops of light that the flash makes up.

## If the budget is constrained, cut in this order

The order follows signal integrity, not cost. Some choices lose information for good; others degrade gently or are simply additions. From most to least worth protecting:

1. **Protect longest:** global shutter, lighting and blur control, stable colour (sensor plus reference card), and raw format/bit-depth. Getting these wrong corrupts the images themselves, with no software fix.
2. **Then detail level.** Lower resolution degrades gently: it limits how finely corrosion can be graded (start with coarser grades, add finer grading with a better sensor later) but leaves detection and segmentation intact.
3. **Then the second camera.** Stereo is an addition. Dropping it loses distance measurement but does not corrupt the images, and it can be added later.

In summary, a lower-resolution global-shutter colour camera with good lighting and raw capture beats a higher-resolution rolling-shutter or auto-white-balanced one. Given a choice, buy down megapixels before shutter, lighting, colour, or format.

## Scope and open decisions

Cameras and illumination only at this stage; localisation and geometry ground-truth sensors come later.

A few trade-offs are TBD. CV constraint to design against for each:

- **Speed vs light:** go as fast as the blur budget allows (under 1 pixel; at target detail, roughly a 30 µs flash). If the flash is not powerful enough, slow down. Fewer sharp frames beat many soft ones.
- **Mono vision vs stereo:** a cost and packaging call. CV wants two for distance measurement; one is a viable MVP.
- **If cost forces video:** use high-bitrate or all-intra encoding, and treat it as the detection tier only. Keep a full-quality stills path (or full-quality keyframes) for anything graded.

The full CV reasoning including detail and tiling maths, the processing pipeline, and compute sizing is in the companion [Compute & Data Pipeline](compute-and-data-pipeline.md) doc.
