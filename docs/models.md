# NVIDIA Alpamayo Models — Comparison & Technical Details

**Author:** Chong Kiat Lim

## Overview

NVIDIA Alpamayo is a family of **Vision-Language-Action (VLA)** models for autonomous driving. These models combine visual perception, language-based reasoning, and trajectory prediction into a unified architecture. They use a novel **Chain-of-Causation (CoC)** reasoning framework that links driving decisions to their causal factors, making the decision-making process interpretable.

Both models are **10B-parameter** models designed for research and development in autonomous driving.

---

## Model Comparison

| Feature | Alpamayo 1 (R1-10B) | Alpamayo 1.5 (10B) |
|---|---|---|
| **Release Date** | December 2025 | March 2026 |
| **VLM Backbone** | Cosmos-Reason | Cosmos-Reason2 |
| **Base Model** | — | Qwen3-VL-8B-Instruct |
| **Parameters** | 10.5B (8.2B + 2.3B) | 10.5B (8.2B + 2.3B) |
| **CoC Reasoning** | ✅ Yes | ✅ Yes (improved) |
| **Trajectory Prediction** | ✅ Yes | ✅ Yes (improved) |
| **Navigation Guidance** | ❌ No | ✅ Yes |
| **Flexible Camera Counts** | ❌ Fixed (4 cameras) | ✅ Configurable |
| **Visual Question Answering** | ❌ No | ✅ Yes |
| **RL Post-Training** | ❌ Weights not released | ✅ Included |
| **Training Data (CoC traces)** | 700K traces | 3M traces |
| **Public Data in Training** | No | Yes (15+ public datasets) |
| **AlpaSim Score** | 0.73 ± 0.01 | 0.81 ± 0.01 |
| **Open-Loop minADE_6** | 1.22m | 1.11m |
| **LingoQA Score** | — | 74.2 |
| **Inference Code** | [github.com/NVlabs/alpamayo](https://github.com/NVlabs/alpamayo) | [github.com/NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5) |
| **HuggingFace** | [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) | [nvidia/Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) |

---

## Architecture

Both models share the same high-level architecture:

```
Multi-Camera Images ─┐
                     ├──▶ VLM Backbone ──▶ CoC Reasoning ──▶ Diffusion Action Expert ──▶ Trajectory
Egomotion History ───┘      (8.2B)          (text output)         (2.3B)                  (64 waypoints)
Text Commands ───────┘
```

### Key Components

1. **VLM Backbone (8.2B params):** Processes multi-camera images and text inputs. Alpamayo 1 uses *Cosmos-Reason*, while v1.5 uses the newer *Cosmos-Reason2* (based on Qwen3-VL-8B).

2. **Chain-of-Causation (CoC) Reasoning:** Generates structured, interpretable reasoning traces that link driving decisions to specific causal factors in the scene (e.g., "The pedestrian is crossing → I need to decelerate → Apply braking").

3. **Diffusion-Based Action Expert (2.3B params):** Converts the VLM's hidden states into trajectory predictions using a diffusion model. Outputs 64 waypoints at 10Hz covering a 6.4-second future horizon.

---

## Alpamayo 1 (R1-10B) — Details

Alpamayo 1 was the initial release, focusing on the core VLA pipeline:

- **Input:** 4 camera views (front-wide, front-tele, cross-left, cross-right) with 0.4s history at 10Hz, plus egomotion history.
- **Output:** Chain-of-Causation reasoning text + 6.4s trajectory.
- **Training:** 80,000 hours of driving data with 700K CoC reasoning traces.
- **Key contribution:** First model to integrate CoC reasoning with trajectory planning for autonomous driving.

### When to use Alpamayo 1

- If you need a simpler, well-established baseline
- If you only have exactly 4 camera inputs matching the expected configuration
- For reproducing results from the original paper (arXiv:2511.00088)

---

## Alpamayo 1.5 (10B) — Details

Alpamayo 1.5 is a significant update that adds interactivity and flexibility:

- **Navigation Guidance:** Accepts navigation instructions (e.g., "turn left at the next intersection") to steer the model's planning.
- **Flexible Camera Counts:** Supports different numbers of cameras, not restricted to exactly 4.
- **Visual Question Answering:** Can answer natural language questions about the driving scene (e.g., "Is it safe to overtake the vehicle ahead?").
- **RL Post-Training:** Model weights include reinforcement learning-based post-training for improved performance.
- **More Training Data:** 3M CoC traces (4x more than v1) plus data from 15+ public driving datasets.

### Inference Methods

Alpamayo 1.5 provides two inference methods:

1. **`sample_trajectories_from_data_with_vlm_rollout`** — Full pipeline: CoC reasoning → trajectory prediction. This is the primary method.
2. **`generate_text`** — Text-only generation for VQA. Returns answers to questions about the scene without trajectory prediction.

### When to use Alpamayo 1.5

- For the best available performance (higher AlpaSim and lower minADE scores)
- When you need navigation-guided planning
- When you want to ask questions about driving scenes (VQA)
- When your camera setup differs from the standard 4-camera config
- **Recommended for most use cases**

---

## Input Specification

Both models accept:

| Input | Format | Details |
|---|---|---|
| **Images** | RGB (1080×1920 px) | Multi-camera, multi-timestep (4 frames at 10Hz per camera). Processor downsamples to 320×576. |
| **Text** | String | User commands, navigation guidance (v1.5), or questions (v1.5 VQA). |
| **Egomotion** | Float: `(x,y,z), R_rot` | 16 waypoints at 10Hz with 3D translation + 9D rotation matrix. |
| **Timestamps** | Float | Required for images and egomotion synchronization. |

## Output Specification

| Output | Format | Details |
|---|---|---|
| **Reasoning** | String | Chain-of-Causation traces explaining driving decisions. |
| **Trajectory** | Float: `(x,y,z), R_rot` | 64 waypoints at 10Hz (6.4s horizon) in ego vehicle frame. Internally uses acceleration/curvature with a unicycle BEV model. |

---

## Hardware Requirements

| Configuration | VRAM Required |
|---|---|
| Single-sample inference (num_traj_samples=1) | ~24 GB |
| Multi-sample inference (num_traj_samples=16) | ~40 GB |
| Multi-sample + CFG (num_traj_samples=16) | ~60 GB |

**Minimum GPU:** NVIDIA RTX 3090 (24GB), RTX 4090, A5000, or equivalent.
**Tested on:** NVIDIA H100 80GB.

---

## Licensing

- **Model weights:** Non-commercial license (commercial licensing available upon request from NVIDIA).
- **Inference code:** Apache License 2.0.

---

## References

- Paper: [Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail](https://arxiv.org/abs/2511.00088)
- NVIDIA Announcement: [NVIDIA Alpamayo at CES 2026](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development)
