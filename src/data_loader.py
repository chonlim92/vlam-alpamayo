"""Dataset utilities for loading driving scene data."""

from datasets import load_dataset
from huggingface_hub import login

from src.config import AppConfig

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry can optionally carry an "hf_id" (Hugging Face dataset repo),
# a "loader" type that selects the loading strategy, and display metadata.

DATASETS = {
    # ── NVIDIA primary datasets ───────────────────────────────────────────
    "physical-ai-av": {
        "hf_id": "nvidia/PhysicalAI-Autonomous-Vehicles",
        "loader": "hf_streaming",
        "description": (
            "1,700 hours of driving data from 25 countries and 2,500+ cities. "
            "306,152 clips (20s each) with multi-camera (7), LiDAR, and radar coverage. "
            "Includes egomotion, calibration, machine labels, and human-verified "
            "Chain-of-Causation reasoning annotations for OOD scenarios."
        ),
        "size": "133 TB (full) — streamable subsets available",
        "url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles",
    },
    "physical-ai-av-nurec": {
        "hf_id": "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
        "loader": "hf_streaming",
        "description": (
            "918 dynamic neural-reconstructed 3D driving scenes (USDZ format) derived "
            "from the PhysicalAI-AV dataset. Used for closed-loop simulation and "
            "evaluation with AlpaSim."
        ),
        "size": "1.77 TB",
        "url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
    },
    # ── Public driving datasets (used in Alpamayo 1.5 training) ───────────
    "drivelm": {
        "hf_id": "OpenDriveLab/DriveLM",
        "loader": "hf_streaming",
        "description": "Driving language-model QA pairs for end-to-end AD.",
        "size": "~2 GB",
        "url": "https://huggingface.co/datasets/OpenDriveLab/DriveLM",
    },
    "lingoqa": {
        "hf_id": "wayveai/LingoQA",
        "loader": "hf_streaming",
        "description": "Language-grounded driving QA benchmark.",
        "size": "~1 GB",
        "url": "https://huggingface.co/datasets/wayveai/LingoQA",
    },
    "nuscenesqa": {
        "hf_id": "qiantianwen/NuScenes-QA",
        "loader": "hf_streaming",
        "description": "Question answering on nuScenes driving scenes.",
        "size": "~500 MB",
        "url": "https://huggingface.co/datasets/qiantianwen/NuScenes-QA",
    },
    "navsim": {
        "hf_id": "autonomousvision/navsim",
        "loader": "hf_streaming",
        "description": "Navigation simulation benchmark for driving.",
        "size": "variable",
        "url": "https://huggingface.co/datasets/autonomousvision/navsim",
    },
    "omnidrive": {
        "hf_id": "NVlabs/OmniDrive",
        "loader": "hf_streaming",
        "description": "Multi-task driving benchmark (3D QA, counterfactual reasoning).",
        "size": "~1 GB",
        "url": "https://huggingface.co/datasets/NVlabs/OmniDrive",
    },
    "talk2car": {
        "hf_id": "talk2car/Talk2Car",
        "loader": "hf_streaming",
        "description": "Natural language grounding in driving scenes.",
        "size": "~500 MB",
        "url": "https://huggingface.co/datasets/talk2car/Talk2Car",
    },
    "coda-lm": {
        "hf_id": "DLCV-BUAA/CODA-LM",
        "loader": "hf_streaming",
        "description": "Corner case analysis for autonomous driving.",
        "size": "~1 GB",
        "url": "https://huggingface.co/datasets/DLCV-BUAA/CODA-LM",
    },
    "drivegpt4": {
        "hf_id": "OpenDriveLab/DriveGPT4",
        "loader": "hf_streaming",
        "description": "Multi-modal driving dialogue dataset.",
        "size": "~2 GB",
        "url": "https://huggingface.co/datasets/OpenDriveLab/DriveGPT4",
    },
    "drive-action": {
        "hf_id": "Drive-Action/Drive-Action",
        "loader": "hf_streaming",
        "description": "Driving action prediction dataset.",
        "size": "variable",
        "url": "https://huggingface.co/datasets/Drive-Action/Drive-Action",
    },
    "maplm": {
        "hf_id": "MapLM/MapLM",
        "loader": "hf_streaming",
        "description": "Map-centric language-model driving dataset.",
        "size": "variable",
        "url": "https://huggingface.co/datasets/MapLM/MapLM",
    },
    "mm-au": {
        "hf_id": "MM-AU/MM-AU",
        "loader": "hf_streaming",
        "description": "Multi-modal action understanding for driving.",
        "size": "variable",
        "url": "https://huggingface.co/datasets/MM-AU/MM-AU",
    },
    "nuinstruct": {
        "hf_id": "nuInstruct/nuInstruct",
        "loader": "hf_streaming",
        "description": "Instruction-following dataset on nuScenes.",
        "size": "variable",
        "url": "https://huggingface.co/datasets/nuInstruct/nuInstruct",
    },
    "senna": {
        "hf_id": "Senna/Senna",
        "loader": "hf_streaming",
        "description": "Driving scene understanding and QA.",
        "size": "variable",
        "url": "https://huggingface.co/datasets/Senna/Senna",
    },
    "roadwork": {
        "hf_id": "Roadwork/Roadwork",
        "loader": "hf_streaming",
        "description": "Construction zone and roadwork scenario data.",
        "size": "variable",
        "url": "https://huggingface.co/datasets/Roadwork/Roadwork",
    },
}

# Ordered list of dataset keys for dropdowns / CLI choices
DATASET_KEYS = list(DATASETS.keys())


def load_sample_data(config: AppConfig, dataset_key: str = "physical-ai-av", num_samples: int = 1):
    """Load sample data from the selected dataset.

    Args:
        config: Application configuration (needs HF token).
        dataset_key: Key from DATASETS registry.
        num_samples: Number of samples to load.

    Returns:
        List of data samples.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Choose from: {DATASET_KEYS}")

    ds_info = DATASETS[dataset_key]
    login(token=config.huggingface_token)

    return _load_hf_streaming(ds_info["hf_id"], num_samples)


def _load_hf_streaming(repo_id: str, num_samples: int) -> list:
    """Load from any HF dataset using streaming mode (no full download)."""
    try:
        ds = load_dataset(repo_id, split="train", streaming=True, trust_remote_code=True)
    except Exception:
        # Some datasets only have a "test" or default split
        ds = load_dataset(repo_id, streaming=True, trust_remote_code=True)
        # Take the first available split
        if hasattr(ds, "keys"):
            first_split = next(iter(ds.keys()))
            ds = ds[first_split]

    samples = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        samples.append(sample)
    return samples


def get_dataset_info() -> list[dict]:
    """Return metadata for all registered datasets."""
    return [
        {"key": key, **{k: v for k, v in info.items() if k not in ("loader", "hf_id")}}
        for key, info in DATASETS.items()
    ]
