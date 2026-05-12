# Usage Guide

**Author:** Chong Kiat Lim

## CLI Usage

The CLI provides four commands: `info`, `run`, `vqa`, and `gui`.

### Show Model & Dataset Info

```bash
python -m src.cli info
```

### Run Reasoning Inference

```bash
# Use default model (from config/.env)
python -m src.cli run

# Specify model explicitly
python -m src.cli run --model alpamayo-1.5

# Process multiple data samples with more trajectory samples
python -m src.cli run --model alpamayo-1.5 --num-data 3 --samples 4

# Use a specific dataset
python -m src.cli run --dataset drivelm
python -m src.cli run --dataset lingoqa --model alpamayo-1.5

# Save JSON to a specific file (video is always saved to output/)
python -m src.cli run --output results/my_result.json
```

Each run generates:
- A **JSON** file with the reasoning trace and trajectory data
- An **annotated MP4 video** (H.264) in the `output/` folder with camera frames, BEV trajectory overlay, reasoning text, and a timeline bar

> **Note:** Video generation requires **ffmpeg** installed at the system level (`sudo apt install ffmpeg`).
> Without it, videos are saved with the mp4v codec which browsers cannot play (shows "NaN:NaN").

### Visual Question Answering (Alpamayo 1.5 only)

```bash
python -m src.cli vqa "What is the ego vehicle doing in this scene?"

# Use a different dataset source
python -m src.cli vqa "Is it safe to change lanes?" --dataset drivelm --output vqa_result.json
```

### Launch the GUI

```bash
# Default (localhost:7860)
python -m src.cli gui

# Custom host/port
python -m src.cli gui --host 0.0.0.0 --port 8080

# Create a public shareable link (via Gradio)
python -m src.cli gui --share
```

---

## GUI Usage

The web GUI (powered by Gradio) has three tabs with a single-page output layout (no sub-tabs for output):

### 1. Reasoning Inference Tab

1. **Select Model** from the dropdown (`alpamayo-1` or `alpamayo-1.5`)
2. Click **Load Model** — wait for confirmation
3. **Select Dataset** from the dropdown (14 datasets available)
4. Set **Samples to Load** and click **Load Data**
5. Adjust **Sample Index** and **Trajectory Samples** count
6. Click **Run Reasoning** to generate output

**Output (single-page, no tabs):**

- **Annotated Video** — Multi-camera composite with:
  - Predicted trajectory (green) and ground-truth trajectory (orange) on camera view
  - BEV mini-map (top-right) with Pred/GT legend
  - Per-frame trajectory info + ADE/FDE metrics (top-left, cyan text)
  - Compact CoC reasoning text (bottom, 2-line scrolling bar)
  - Timeline progress bar
- **Trajectory (BEV)** — Zoomable, full-screen capable Bird's-Eye-View plot showing predicted vs GT trajectories with ADE/FDE. Supports pinch-zoom and fullscreen button.
- **Metrics** — minADE and minFDE values displayed side-by-side with the trajectory plot, plus a summary markdown table.

**Understanding the parameters:**

| Parameter | What it controls |
|---|---|
| **Samples to Load** | Number of **driving scenes/clips** to download. Each is a distinct scenario (different road, timestamp). |
| **Trajectory Samples** | Number of **predicted future paths** the model generates per scene. More = better minADE/minFDE, higher VRAM. E.g., 8 → the model predicts 8 possible futures, minADE picks the best. |
| **Sample Index** | Which loaded scene to run inference on (0-indexed). |

### 2. Visual QA Tab (Alpamayo 1.5 only)

1. Ensure Alpamayo 1.5 is loaded (from the Reasoning tab)
2. Select a **Sample Index** (clamped to valid range automatically)
3. Type your question
4. Click **Ask Question**

### 3. Reference Tab

Displays model comparison table and the full list of 14 supported datasets.

---

## Output Format

Results are saved as JSON files in the `output/` directory:

```json
{
  "model": "Alpamayo 1.5 (10B)",
  "reasoning_trace": "The ego vehicle is approaching an intersection...",
  "trajectory": {
    "waypoints": [[x, y, z, ...], ...],
    "horizon_seconds": 6.4,
    "frequency_hz": 10,
    "num_waypoints": 64
  },
  "timestamp": "2026-05-12T10:30:00"
}
```

---

## Configuration

Copy `config/.env.sample` to `config/.env` and configure:

| Variable | Default | Description |
|---|---|---|
| `HUGGINGFACE_API_TOKEN` | (required) | Your Hugging Face access token |
| `DEFAULT_MODEL` | `alpamayo-1.5` | Model to use by default |
| `DEVICE` | `cuda` | PyTorch device |
| `DTYPE` | `bfloat16` | Model precision |
| `ATTN_IMPLEMENTATION` | `sdpa` | Attention backend (`flash_attention_2` or `sdpa`) |
| `NUM_TRAJ_SAMPLES` | `1` | Number of trajectory samples |
| `OUTPUT_DIR` | `output` | Directory for result files |
| `GUI_HOST` | `127.0.0.1` | GUI server host |
| `GUI_PORT` | `7860` | GUI server port |

`config/.env` is git-ignored (your real token stays local). Environment variables override file values.
