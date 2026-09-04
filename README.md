# ComfyUI-YCNodes-MiniMax-H3

A ComfyUI node package specifically designed for the MiniMax H3 video model, containing 6 nodes that cover secondary sampling condition injection, time-segmented prompt control, attention receptive field constraints, dynamic CFG scheduling, low-noise detail refinement, and spatial tiled sampling.

No third-party dependencies required, only PyTorch.

Recommended online one-click ComfyUI platform:
[RunningHub China](https://www.runninghub.cn?inviteCode=cn-v1079) ---- [RunningHub International](https://www.runninghub.ai?inviteCode=rh-v1091) - Rich plugins and models, ready to use
---

## Node Overview

| Node | Category | Function |
|------|------|------|
| MiniMax H3 Image to Video (Tail) | conditioning | Secondary sampling condition node, can pass through first-sampling latent for continued sampling |
| H3 Prompt Relay | conditioning | Time-segmented prompt control, different prompts for different time periods |
| H3 Distance Attention Patcher | Attention | Spatiotemporal Gaussian receptive field mask, prevents background from assimilating local details |
| H3 Dynamic CFG Scheduler | scheduler | Dynamically adjusts CFG guidance strength based on denoising stage |
| H3 Sigma Refiner | scheduler | Local step addition in low-noise interval, eliminates motion edge pixel artifacts |
| H3 Tiled Sampler | sampling | Tiled sampling to reduce VRAM usage, adapted from [10S-Comfy-nodes](https://github.com/TenStrip/10S-Comfy-nodes) LTX Tiled Sampler |

---

## 1. MiniMax H3 Image to Video (Tail) (Secondary Sampling Condition Node)

**Principle:** The official `MiniMaxH3ImageToVideo` always outputs empty AV latent (pure noise starting point), which cannot be used for secondary sampling continuation. This node adds a new `video_latent` input: when the first-sampling output latent is passed in, it is directly passed through, and keyframe anchors are recalculated according to the actual frame count, enabling posterior sampling/high-definition refinement.

**Secondary HD Usage:** Connect the first-sampling output LATENT to `video_latent`, first/last frames are optional, `width`/`height` automatically align with the second-sampling latent resolution. Set `apply_keyframes` to `disable` to skip keyframe injection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip` | CLIP | - | H3 CLIP model |
| `vae` | VAE | - | H3 VAE |
| `prompt` | STRING | - | Prompt |
| `width` | INT | 1344 | Secondary sampling target width (automatically aligns with video_latent) |
| `height` | INT | 768 | Secondary sampling target height (automatically aligns with video_latent) |
| `length` | INT | 124 | Frame count, determines empty latent when video_latent is not passed |
| `first_frame` | IMAGE | Optional | First frame image |
| `last_frame` | IMAGE | Optional | Last frame image |
| `video_latent` | LATENT | Optional | First-sampling output latent, passes through and recalculates anchors when provided |
| `apply_keyframes` | COMBO | enable | enable / disable, skips keyframe injection when disabled |

---

## 2. H3 Prompt Relay (Time-Segmented Prompt Control)

**Principle:** H3 uses packed self-attention (text + cond + audio + video in the same sequence), there is no independent cross-attention. This node applies temporal penalty masks to the video query -> text key path in the self-attention matrix, achieving the effect where different time periods only attend to corresponding prompts.

**Usage:** Copy the official prompt verbatim to `local_prompts`, add `|` delimiter at segmentation points. `global_prompt` (optional) fills in global style tone, visible to all frames throughout. Supports official image-to-video node (`MiniMaxH3ImageToVideo`) and multi-parameter version (`MiniMaxH3ImageToVideoMultiParams`).

```
CLIP -> [MiniMaxH3ImageToVideo / MultiParams] -> CONDITIONING ─┐
CLIP ──────────────────────────────────────────────────────────┤
latent ──────────────────────────────────────────────────────> [H3PromptRelay] -> MODEL -> [Sampler]
                                                                ↑                   CONDITIONING
                                                                └─── (pass-through) ─────────┘
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | - | H3 model |
| `conditioning` | CONDITIONING | - | Conditioning output from official node |
| `clip` | CLIP | - | H3 CLIP model |
| `latent` | LATENT | - | H3 video latent |
| `global_prompt` | STRING | Empty | Global style/theme tone, visible to all frames throughout (optional) |
| `local_prompts` | STRING | Empty | Segmented prompts, separated by `\|` |
| `segment_lengths` | STRING | Empty | Comma-separated pixel frame counts, auto-divided if left empty |
| `epsilon` | FLOAT | 0.001 | Penalty decay parameter, smaller values create sharper boundaries |
| `patch_ratio` | FLOAT | 1.0 | 0.0-1.0, proportion of DiT Blocks to patch. Recommend 0.3-0.7 when audio is present |

---

## 3. H3 Distance Attention Patcher (Distance Attention Constraint)

**Principle:** Addresses the issue where limbs and faces are easily assimilated or torn apart by the background in panoramic scenes. Forces the model to constrain local attention during mid-early denoising through spatiotemporal Gaussian receptive field masks, blocking large-area background feature assimilation of tiny details.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | H3 model |
| `receptive_field_scale` | FLOAT | 15.0 | 1.0 ~ 100.0 | Receptive field scale, smaller = more local |
| `temporal_weight` | FLOAT | 2.0 | 0.1 ~ 10.0 | Distance weight of temporal axis relative to spatial axis |
| `start_at_sigma` | FLOAT | 4.5 | 0.0 ~ 20.0 | Sigma threshold to start constraint |
| `end_at_sigma` | FLOAT | 0.0 | 0.0 ~ 5.0 | Sigma threshold to end constraint |
| `num_frames` | INT | 17 | 1 ~ 256 | Total video frames |
| `original_width` | INT | 864 | 128 ~ 2048 | Video width |
| `original_height` | INT | 480 | 128 ~ 2048 | Video height |

---

## 4. H3 Dynamic CFG Scheduler (Dynamic CFG Scheduling)

**Principle:** Dynamically adjusts CFG guidance strength based on denoising stage. Low CFG at high sigma (early composition) preserves overall shape, high CFG at low sigma (late details) enhances details. H3 flow matching defaults to CFG=1.0, dynamic range is very small, fine-tuning is sufficient.

**Connection:** Insert between MODEL and sampler.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | H3 model |
| `cfg_low` | FLOAT | 0.9 | 0.5 ~ 2.0 | CFG at high sigma (early composition) |
| `cfg_high` | FLOAT | 1.1 | 0.5 ~ 2.0 | CFG at low sigma (late details) |
| `start_at_sigma` | FLOAT | 3.0 | 0.0 ~ 20.0 | Sigma threshold to start dynamic scheduling |
| `end_at_sigma` | FLOAT | 0.0 | 0.0 ~ 5.0 | Sigma threshold to end dynamic scheduling |

---

## 5. H3 Sigma Refiner (Low-Noise Detail Refinement)

**Principle:** Local step addition in low Sigma interval — keeps the high-noise head of the original schedule unchanged, resamples the tail from the threshold point into a longer, smoother curve, allowing the model to take more steps during the detail finishing phase, eliminating mosaics and pixel chaos at high-motion edges.

**Connection:** Insert between scheduler and sampler.

```
BasicScheduler -> (sigmas) -> H3 Sigma Refiner -> (sigmas) -> SamplerCustomAdvanced
```

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sigmas` | SIGMAS | - | - | Original noise sequence |
| `extra_steps` | INT | 1 | 0 ~ 15 | Extra steps added in low-noise interval |
| `start_at_sigma` | FLOAT | 0.7 | 0.0 ~ 20.0 | Sigma threshold to activate step addition |
| `end_at_sigma` | FLOAT | 0.0 | 0.0 ~ 5.0 | Sigma boundary to end refinement |
| `spacing` | COMBO | cosine | cosine / linear / exponential | Tail interpolation distribution curve |

**spacing curves:**
- **cosine** (default): Denser distribution approaching 0, smoothest noise elimination.
- **linear**: Uniform distribution.
- **exponential**: Energy shifts forward, large steps toward the end point.

---

## 6. H3 Tiled Sampler (Spatial Tiled Sampling)

**Principle:** During 768p/2K high-resolution upsampling refinement, H3 DiT packed self-attention token count far exceeds training distribution, causing VRAM explosion or quality degradation. This node tiles along H or W axis spatially, samples each tile independently then fuses with cosine window, strictly maintaining H3 video/audio independent modalities, audio passthrough does not participate in sampling.

**Connection:** Replace `SamplerCustomAdvanced`, connect to noise / guider / sampler / sigmas / latent_image.

**Applicable Scenarios:** Secondary HD refinement (low-noise starting point), not suitable for heavy denoising (pure noise starting point will destroy global consistency).

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `noise` | NOISE | - | - | H3 noise generator |
| `guider` | GUIDER | - | - | H3 CFG/STG guider |
| `sampler` | SAMPLER | - | - | Sampling algorithm |
| `sigmas` | SIGMAS | - | - | H3 noise schedule |
| `latent_image` | LATENT | - | - | H3 video latent |
| `bypass_tiling` | BOOLEAN | False | - | When True, single sampling, equivalent to SamplerCustomAdvanced |
| `tile_axis` | COMBO | auto | auto / H / W | Tiling axis, auto takes the longer axis |
| `n_tiles` | INT | 2 | 1 ~ 8 | Number of tiles, 1 equals bypass |
| `tile_overlap` | INT | 8 | 0 ~ 32 | Overlapping token count between adjacent blocks in latent domain |
| `max_size_for_no_tile` | INT | 24 | 8 ~ 256 | Auto bypass when target axis ≤ this value |
| `target_frames` | INT | 17 | 1 ~ 512 | Minimum frame protection (non-truncation), pads if insufficient |
| `frame_padding_mode` | COMBO | replicate_last | replicate_last / zero / error | Padding method when frame count is insufficient |
| `debug` | BOOLEAN | False | - | Print shape and value range for each tile |

---

## Installation

1. Place the `ComfyUI-YCNodes-MiniMax-H3` directory into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Search for `H3` in the node panel to find all nodes.

## Recommended Workflow Connections

**First Sampling (Low Resolution):**
```
[CLIP] ─────────────────────────────────────────────────────────┐
[Image] -> [MiniMaxH3ImageToVideo / MultiParams] -> COND ────────┤
                                                                ├─> [H3PromptRelay] -> MODEL ─┐
[CLIP] ─────────────────────────────────────────────────────────┘                            │
                                                                                             ├─> [H3DynamicCFGScheduler] -> MODEL -> [Sampler]
[BasicScheduler] -> [H3SigmaRefiner] -> SIGMAS ──────────────────────────────────────────────┘
```

**Second Sampling (HD Refinement, Tail Node Continuation, Tiled Sampling):**
```
First-Sampling LATENT -> video_latent ─┐
[Image] -> [MiniMaxH3ImageToVideoTail] -> COND ─┐
[CLIP] ────────────────────────────────────────┤
                                                ├─> [H3PromptRelay] -> MODEL -> [H3DynamicCFGScheduler] -> MODEL ─┐
                                                LATENT ───────────────────────────────────────────────────────────┤
                                                                                                                   ├─> [H3TiledSampler] -> LATENT
[BasicScheduler] -> [H3SigmaRefiner] -> SIGMAS ────────────────────────────────────────────────────────────────────┘
```
