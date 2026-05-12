"""Dataset utilities for loading driving scene data."""

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, login

from src.config import AppConfig

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry can optionally carry an "hf_id" (Hugging Face dataset repo),
# a "loader" type that selects the loading strategy, and display metadata.

DATASETS = {
    # ── NVIDIA primary datasets ───────────────────────────────────────────
    "physical-ai-av": {
        "hf_id": "nvidia/PhysicalAI-Autonomous-Vehicles",
        "loader": "physical_ai_av_sdk",
        "description": (
            "1,700 hours of driving data from 25 countries and 2,500+ cities. "
            "306,152 clips (20s each) with multi-camera (7), LiDAR, and radar coverage. "
            "Includes egomotion, calibration, machine labels, and human-verified "
            "Chain-of-Causation reasoning annotations for OOD scenarios."
        ),
        "size": "133 TB (full) — downloads individual clips via SDK",
        "url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles",
    },
    "physical-ai-av-nurec": {
        "hf_id": "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
        "loader": "hf_parquet",
        "parquet_path": None,
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

    loader = ds_info["loader"]
    if loader == "physical_ai_av_sdk":
        return _load_physical_ai_av_sdk(config, num_samples)
    if loader == "hf_parquet":
        return _load_hf_parquet(
            ds_info["hf_id"], ds_info.get("parquet_path"), num_samples,
        )
    return _load_hf_streaming(ds_info["hf_id"], num_samples)


# ── PhysicalAI-AV SDK loader ─────────────────────────────────────────────────

def _load_physical_ai_av_sdk(config: AppConfig, num_samples: int) -> list:
    """Load driving clips with camera video via the physical_ai_av SDK.

    Downloads the OOD reasoning parquet to find interesting clips, then uses
    the SDK to download actual camera data for those clips.  Returns samples
    with numpy video frames that the visualization pipeline can render.
    """
    try:
        from physical_ai_av import PhysicalAIAVDatasetInterface
    except ImportError:
        raise ImportError(
            "physical_ai_av SDK not installed. Run: pip install physical_ai_av\n"
            "Alternatively, select a different dataset."
        )

    print("Initializing PhysicalAI-AV SDK...")
    ds = PhysicalAIAVDatasetInterface(
        token=config.huggingface_token,
        confirm_download_threshold_gb=float("inf"),  # don't block GUI
    )

    # Load reasoning parquet to find OOD clips with CoC annotations
    reasoning_df = _load_reasoning_index(ds)

    # Pick clip IDs — prefer OOD-annotated clips, fall back to clip_index
    clip_ids = _select_clip_ids(ds, reasoning_df, num_samples)

    camera_feature = "camera_front_wide_120fov"
    samples = []

    for i, clip_id in enumerate(clip_ids):
        print(f"[{i + 1}/{len(clip_ids)}] Downloading clip {clip_id}...")

        sample = {"clip_id": clip_id, "source": "physical_ai_av_sdk"}

        # Download and extract camera frames
        try:
            ds.download_clip_features(clip_id, features=[camera_feature])
            video_reader = ds.get_clip_feature(clip_id, camera_feature)
            frames = _extract_video_frames(video_reader, max_frames=40)
            if frames:
                sample["camera_front_wide_120fov"] = frames
                print(f"  Camera: {len(frames)} frames extracted")
            else:
                print("  Camera: no frames extracted (reader returned empty)")
        except Exception as e:
            print(f"  Camera download failed: {e}")

        # Download egomotion
        try:
            ds.download_clip_features(clip_id, features=["egomotion"])
            ego = ds.get_clip_feature(clip_id, "egomotion")
            sample["egomotion"] = ego
            print("  Egomotion: loaded")
        except Exception as e:
            print(f"  Egomotion: {e}")

        # Attach reasoning annotations if available
        if reasoning_df is not None and clip_id in reasoning_df.index:
            row = reasoning_df.loc[clip_id]
            sample["event_cluster"] = row.get("event_cluster", "")
            sample["events"] = row.get("events", [])
            sample["split"] = row.get("split", "")

        samples.append(sample)

    if not samples:
        raise RuntimeError("No clips could be loaded. Check dataset access and network.")

    return samples


def _load_reasoning_index(ds) -> pd.DataFrame | None:
    """Download the OOD reasoning parquet and return it indexed by clip_id."""
    try:
        path = ds.download_file("reasoning/ood_reasoning.parquet")
        df = pd.read_parquet(path)
        # If clip ID is in the index, keep it; otherwise check columns
        if df.index.name and "clip" in str(df.index.name).lower():
            return df
        # Check if there's a clip column we can use as index
        for col in df.columns:
            if "clip" in col.lower() or "uuid" in col.lower():
                return df.set_index(col)
        # No clip ID found — can't correlate with SDK clips
        print("  Reasoning parquet has no clip ID column — annotations won't be linked")
        return None
    except Exception as e:
        print(f"  Could not load reasoning parquet: {e}")
        return None


def _select_clip_ids(ds, reasoning_df: pd.DataFrame | None, num_samples: int) -> list:
    """Select clip IDs to download — prefer OOD-annotated clips."""
    if reasoning_df is not None and len(reasoning_df) > 0:
        # Use clips that have OOD reasoning annotations
        ids = reasoning_df.index[:num_samples].tolist()
        if ids:
            print(f"Selected {len(ids)} OOD-annotated clip(s)")
            return ids

    # Fall back to first N clips from the dataset clip index
    ids = ds.clip_index.index[:num_samples].tolist()
    print(f"Selected {len(ids)} clip(s) from clip index")
    return ids


def _extract_video_frames(video_reader, max_frames: int = 40) -> list[np.ndarray]:
    """Extract numpy BGR frames from a SeekVideoReader or similar object."""
    frames = []

    # Try iterating (SeekVideoReader should be iterable)
    try:
        for frame in video_reader:
            arr = _frame_to_numpy(frame)
            if arr is not None:
                frames.append(arr)
            if len(frames) >= max_frames:
                break
        if frames:
            return frames
    except (TypeError, StopIteration, AttributeError):
        pass

    # Try indexing
    try:
        length = len(video_reader)
        step = max(1, length // max_frames)
        for i in range(0, length, step):
            arr = _frame_to_numpy(video_reader[i])
            if arr is not None:
                frames.append(arr)
            if len(frames) >= max_frames:
                break
        if frames:
            return frames
    except (TypeError, IndexError, AttributeError):
        pass

    # Try .read() method
    try:
        if hasattr(video_reader, "read"):
            while len(frames) < max_frames:
                ret = video_reader.read()
                if ret is None:
                    break
                arr = _frame_to_numpy(ret)
                if arr is not None:
                    frames.append(arr)
    except Exception:
        pass

    return frames


def _frame_to_numpy(frame) -> np.ndarray | None:
    """Convert various frame types to a BGR numpy array."""
    if frame is None:
        return None

    # Already numpy
    if isinstance(frame, np.ndarray):
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        if frame.ndim == 2:
            import cv2
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    # PIL Image
    try:
        from PIL import Image
        if isinstance(frame, Image.Image):
            import cv2
            arr = np.array(frame)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
    except ImportError:
        pass

    # PyAV VideoFrame
    try:
        if hasattr(frame, "to_ndarray"):
            return frame.to_ndarray(format="bgr24")
    except Exception:
        pass

    # Torch tensor
    try:
        import torch
        if isinstance(frame, torch.Tensor):
            arr = frame.cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            return arr.astype(np.uint8)
    except ImportError:
        pass

    return None


# ── Other loaders ─────────────────────────────────────────────────────────────


def _load_hf_parquet(repo_id: str, parquet_path: str | None, num_samples: int) -> list:
    """Load samples from a specific parquet file in an HF dataset repo.

    Used for large NVIDIA datasets that don't follow standard HF dataset format.
    """
    if parquet_path is None:
        raise ValueError(
            f"Dataset '{repo_id}' does not have a directly loadable parquet file. "
            "Use the physical_ai_av SDK or download specific chunks manually."
        )

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=parquet_path,
        repo_type="dataset",
    )
    df = pd.read_parquet(local_path)
    rows = df.head(num_samples)
    return [row.to_dict() for _, row in rows.iterrows()]


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
