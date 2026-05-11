"""Dataset utilities for loading PhysicalAI-AV data."""

from huggingface_hub import login
from physical_ai_av import PhysicalAIAVDataset

from src.config import AppConfig

# Companion datasets
DATASETS = {
    "physical-ai-av": {
        "repo_id": "nvidia/PhysicalAI-Autonomous-Vehicles",
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
        "repo_id": "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
        "description": (
            "918 dynamic neural-reconstructed 3D driving scenes (USDZ format) derived "
            "from the PhysicalAI-AV dataset. Used for closed-loop simulation and "
            "evaluation with AlpaSim."
        ),
        "size": "1.77 TB",
        "url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
    },
}


def load_sample_data(config: AppConfig, num_samples: int = 1):
    """Load sample data from the PhysicalAI-AV dataset.

    Streams data directly from Hugging Face; no full download required.

    Args:
        config: Application configuration (needs HF token).
        num_samples: Number of samples to load.

    Returns:
        List of data samples ready for inference.
    """
    login(token=config.huggingface_token)

    dataset = PhysicalAIAVDataset(
        repo_id=DATASETS["physical-ai-av"]["repo_id"],
    )

    samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(sample)

    return samples


def get_dataset_info() -> list[dict]:
    """Return metadata for available datasets."""
    return [
        {"key": key, **info}
        for key, info in DATASETS.items()
    ]
