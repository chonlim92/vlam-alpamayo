# Datasets for VLAM-Alpamayo

**Author:** Chong Kiat Lim

## Recommended Datasets

Both Alpamayo models are designed to work with NVIDIA's PhysicalAI Autonomous Vehicles datasets.

---

### 1. PhysicalAI-Autonomous-Vehicles (Primary Dataset)

| | |
|---|---|
| **URL** | [huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |
| **Size** | 133 TB (full dataset — streamable subsets available) |
| **Clips** | 306,152 clips, each 20 seconds long |
| **Duration** | 1,700 hours of driving |
| **Geography** | 25 countries, 2,500+ cities |
| **License** | NVIDIA Autonomous Vehicle Dataset License (commercial/non-commercial AV use) |

#### Sensors

- **7 cameras:** front-wide (120°), front-tele (30°), cross-left (120°), cross-right (120°), rear-left (70°), rear-right (70°), rear-tele (30°)
- **LiDAR:** top 360° rotating — 298,326 clips
- **Radar:** up to 10 radars — 160,761 clips

#### Labels & Annotations

- Egomotion (online and offline-optimized)
- Obstacle detection (offline)
- Calibration (camera intrinsics, sensor extrinsics)
- **Chain-of-Causation (CoC) reasoning labels** — human-verified for OOD driving scenarios (1,450 train + 290 val)

#### Diversity

- **Traffic:** no traffic, light, medium, heavy
- **Roads:** highways, urban, residential, rural
- **Weather:** clear, rain, snow, fog
- **Surface:** dry, wet, snow/ice
- **Time:** daytime, nighttime
- **Infrastructure:** tunnels, bridges, roundabouts, railway crossings, toll booths, etc.

#### How to Access

1. Create a [Hugging Face account](https://huggingface.co/join)
2. Accept the NVIDIA Autonomous Vehicle Dataset License Agreement on the dataset page
3. Create a [User Access Token](https://huggingface.co/settings/tokens)
4. Install the developer toolkit:
   ```bash
   pip install physical_ai_av
   ```

> **Note:** You do NOT need to download the entire 133TB dataset. The `physical_ai_av` package supports streaming subsets directly from Hugging Face.

---

### 2. PhysicalAI-Autonomous-Vehicles-NuRec (Simulation Dataset)

| | |
|---|---|
| **URL** | [huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) |
| **Size** | 1.77 TB |
| **Scenes** | 918 neural-reconstructed 3D driving scenes |
| **Format** | USDZ files with surface meshes |
| **License** | NVIDIA Autonomous Vehicle Dataset License |

This dataset contains **dynamic 3D neural reconstructions** of driving scenes, generated using NVIDIA NuRec from 6 camera views. Each scene is ~20 seconds long and can be used for:

- **Closed-loop simulation** with [AlpaSim](https://github.com/NVlabs/alpasim)
- 3D scene visualization and analysis
- SFT and RL training workflows

#### Scene Labels

Each scene includes labels for:
- Behavior: driving_straight, stop, lane_change, turns, reverse
- Layout: intersection, roundabout, construction_zone, parking_lot, etc.
- Weather, lighting, road type, surface conditions, traffic density, VRU presence

---

## Other Public Datasets Used in Alpamayo 1.5 Training

Alpamayo 1.5 incorporated data from these public driving datasets during training, which may also be useful for evaluation:

| Dataset | Focus |
|---|---|
| [DriveLM](https://github.com/OpenDriveLab/DriveLM) | Driving language-model QA pairs |
| [LingoQA](https://github.com/wayveai/LingoQA) | Language-grounded driving QA benchmark |
| [nuScenesQA](https://github.com/qiantianwen/NuScenes-QA) | Question answering on nuScenes |
| [NAVSIM](https://github.com/autonomousvision/navsim) | Navigation simulation benchmark |
| [Omnidrive](https://github.com/NVlabs/OmniDrive) | Multi-task driving benchmark |
| [Talk2Car](https://github.com/talk2car/Talk2Car) | Natural language grounding in driving |
| [CODA-LM](https://github.com/DLCV-BUAA/CODA-LM) | Corner case analysis |
| [DriveGPT4](https://github.com/OpenDriveLab/DriveGPT4) | Multi-modal driving dialogue |
| Drive-Action | Driving action prediction |
| MapLM | Map-centric language-model driving |
| MM-AU | Multi-modal action understanding for driving |
| nuInstruct | Instruction-following on nuScenes |
| Senna | Driving scene understanding and QA |
| Roadwork | Construction zone and roadwork scenarios |

All of these datasets are selectable via the CLI (`--dataset`) and the GUI dataset dropdown. The app streams data from Hugging Face without requiring a full download.

---

## Download Examples

### Stream data with physical_ai_av (recommended)

```python
from physical_ai_av import PhysicalAIAVDataset

dataset = PhysicalAIAVDataset(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
)

for sample in dataset:
    # sample contains multi-camera images, egomotion, timestamps
    print(sample.keys())
    break
```

### Download NuRec scenes

```python
from huggingface_hub import login, snapshot_download
import os

login(token=os.getenv("HUGGINGFACE_API_TOKEN"))

# Download a specific scene
snapshot_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
    repo_type="dataset",
    allow_patterns="sample_set/25.07_release/Batch0002/001b28cb-*/*",
)
```
