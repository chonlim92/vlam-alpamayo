# 🏔️ VLAM-Alpamayo

**Author:** Chong Kiat Lim

An application for autonomous driving reasoning and explanation using NVIDIA's Alpamayo Vision-Language-Action (VLA) models. Provides both a **CLI** and a **web GUI** (Gradio) to run Chain-of-Causation reasoning and trajectory prediction on driving scenes.

## Supported Models

| Model | Key | Description |
|---|---|---|
| [Alpamayo 1 (R1-10B)](https://huggingface.co/nvidia/Alpamayo-R1-10B) | `alpamayo-1` | Core CoC reasoning + trajectory prediction |
| [Alpamayo 1.5 (10B)](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | `alpamayo-1.5` | Adds navigation guidance, flexible cameras, VQA, RL post-training **(recommended)** |

For a detailed comparison, see [docs/models.md](docs/models.md).

## Dataset

This project supports **14 driving datasets** streamed on-demand from Hugging Face (no full download required), including NVIDIA's [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (1,700 hours, 306K clips, 25 countries), plus DriveLM, LingoQA, nuScenesQA, NAVSIM, Omnidrive, Talk2Car, CODA-LM, DriveGPT4, and more. See [docs/datasets.md](docs/datasets.md) for the full list.

---

## Prerequisites

### Hardware

- **GPU:** NVIDIA GPU with **≥ 24 GB VRAM** (e.g., RTX 3090, RTX 4090, A5000, H100)
- Multi-sample inference (16 samples) requires ~40 GB; with CFG ~60 GB

### Software

| Requirement | Version |
|---|---|
| **OS** | Linux (recommended; tested). Windows/WSL2 may work but is unverified. |
| **Python** | 3.12.x |
| **PyTorch** | ≥ 2.8 with CUDA support |
| **CUDA Toolkit** | 12.x (for Flash Attention 2; optional if using SDPA fallback) |
| **Transformers** | ≥ 4.57.1 |
| **DeepSpeed** | ≥ 0.17.4 |

### Hugging Face Access

You **must** request access to these gated resources before using the models:

1. [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) (Alpamayo 1 weights)
2. [nvidia/Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) (Alpamayo 1.5 weights)
3. [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (dataset)

Get your token at: https://huggingface.co/settings/tokens

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd vlam-alpamayo
```

### 2. Create and activate a virtual environment

```bash
python -m venv .alpamayo_venv

# Linux/macOS
source .alpamayo_venv/bin/activate

# Windows
.\.alpamayo_venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the Alpamayo model packages

```bash
# Alpamayo 1
pip install git+https://github.com/NVlabs/alpamayo.git

# Alpamayo 1.5
pip install git+https://github.com/NVlabs/alpamayo1.5.git
```

### 5. (Optional) Install Flash Attention 2

Requires CUDA Toolkit with `nvcc` on your PATH:

```bash
pip install flash-attn --no-build-isolation
```

If you skip this, the app defaults to PyTorch's built-in SDPA attention (configured in `config/.env`).

### 6. Configure your Hugging Face token

Edit `config/.env` and set your token:

```
HUGGINGFACE_API_TOKEN=hf_your_token_here
```

Or set it as an environment variable:

```bash
export HUGGINGFACE_API_TOKEN=hf_your_token_here
# or
export HF_TOKEN=hf_your_token_here
```

---

## Quick Start

### CLI — Show info

```bash
python -m src.cli info
```

### CLI — Run reasoning inference

```bash
# Use the default model (Alpamayo 1.5)
python -m src.cli run

# Use Alpamayo 1 with 3 samples from a specific dataset
python -m src.cli run --model alpamayo-1 --num-data 3 --dataset drivelm

# Save JSON to a specific file (video is always saved to output/)
python -m src.cli run --output results/my_result.json
```

Each run generates a **JSON result file** and an **annotated MP4 video** (with camera frames, BEV trajectory overlay, reasoning text, and timeline) in `output/`.

### CLI — Visual Question Answering (Alpamayo 1.5 only)

```bash
python -m src.cli vqa "What is the ego vehicle doing?"
python -m src.cli vqa "Is it safe to change lanes?" --dataset lingoqa
```

### GUI — Launch the web interface

```bash
python -m src.cli gui
```

Then open http://127.0.0.1:7860 in your browser.

```bash
# Share publicly via Gradio
python -m src.cli gui --share
```

For full usage details, see [docs/usage.md](docs/usage.md).

---

## Project Structure

```
vlam-alpamayo/
├── config/
│   └── .env                  # Configuration (HF token, model settings)
├── docs/
│   ├── models.md             # Model comparison & technical details
│   ├── datasets.md           # Dataset documentation
│   └── usage.md              # Detailed usage guide
├── src/
│   ├── __init__.py
│   ├── cli.py                # CLI application
│   ├── gui.py                # Gradio web GUI
│   ├── config.py             # Configuration loader
│   ├── model_loader.py       # Model loading & authentication
│   ├── data_loader.py        # Dataset loading (14 datasets)
│   ├── inference.py          # Inference engine
│   └── visualization.py      # Video & trajectory rendering
├── output/                   # Generated results (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Configuration

All settings are in `config/.env`. Environment variables override file values.

| Variable | Default | Description |
|---|---|---|
| `HUGGINGFACE_API_TOKEN` | *(required)* | Hugging Face access token |
| `DEFAULT_MODEL` | `alpamayo-1.5` | Default model (`alpamayo-1` or `alpamayo-1.5`) |
| `DEVICE` | `cuda` | PyTorch device |
| `DTYPE` | `bfloat16` | Model precision |
| `ATTN_IMPLEMENTATION` | `sdpa` | `flash_attention_2` or `sdpa` |
| `NUM_TRAJ_SAMPLES` | `1` | Trajectory samples per inference |
| `OUTPUT_DIR` | `output` | Result output directory |
| `GUI_HOST` | `127.0.0.1` | GUI server host |
| `GUI_PORT` | `7860` | GUI server port |

---

## Troubleshooting

### CUDA out-of-memory

- Ensure your GPU has ≥ 24 GB VRAM
- Reduce `NUM_TRAJ_SAMPLES` in `config/.env`
- Close other GPU-intensive applications

### Flash Attention build errors

If `flash-attn` fails to install, use SDPA instead (the default in `config/.env`):

```
ATTN_IMPLEMENTATION=sdpa
```

### Model download is slow

The model weights are ~22 GB. On a 100 MB/s connection, expect ~2.5 minutes. Weights are cached after the first download.

---

## License

- **This application:** MIT
- **Alpamayo model weights:** [Non-commercial license](https://huggingface.co/nvidia/Alpamayo-R1-10B/blob/main/LICENSE) (commercial licensing available from NVIDIA)
- **Alpamayo inference code:** Apache License 2.0
- **PhysicalAI-AV dataset:** [NVIDIA AV Dataset License](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/blob/main/LICENSE.pdf)

## References

- [Alpamayo-R1 Paper (arXiv:2511.00088)](https://arxiv.org/abs/2511.00088)
- [NVIDIA Alpamayo at CES 2026](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development)
- [Alpamayo 1 Code](https://github.com/NVlabs/alpamayo)
- [Alpamayo 1.5 Code](https://github.com/NVlabs/alpamayo1.5)
- [AlpaSim Simulator](https://github.com/NVlabs/alpasim)

---

## Tech Stack

| Technology | Description |
|---|---|
| **Python 3.12** | Core language for the application |
| **PyTorch ≥ 2.8** | Deep learning framework for model inference and tensor operations |
| **Hugging Face Transformers** | Model loading, tokenization, and inference pipeline for 10B-parameter VLA models |
| **Hugging Face Hub / Datasets** | Streaming access to gated model weights and 14 driving datasets without full download |
| **NVIDIA Alpamayo (VLA)** | Vision-Language-Action models combining perception, Chain-of-Causation reasoning, and diffusion-based trajectory prediction |
| **DeepSpeed** | Distributed inference optimization for large-scale model loading |
| **Flash Attention 2 / SDPA** | Memory-efficient attention implementations for running 10B models on consumer GPUs (24 GB+) |
| **Gradio** | Interactive web GUI with video playback, image display, dropdowns, sliders, and tabbed layout |
| **OpenCV (cv2)** | Video encoding (MP4), frame compositing, BEV trajectory overlay, and text rendering |
| **NumPy** | Numerical operations for trajectory processing, coordinate transforms, and visualization |
| **argparse** | CLI framework with subcommands (`run`, `vqa`, `gui`, `info`), dataset/model selection |
| **python-dotenv** | Configuration management via `config/.env` with environment variable overrides |
| **ffmpeg** | Optional H.264 re-encoding for browser-compatible video playback |
| **Git / GitHub** | Version control and collaborative development |
