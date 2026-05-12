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
        "loader": "unsupported",
        "description": (
            "918 dynamic neural-reconstructed 3D driving scenes (USDZ format) derived "
            "from the PhysicalAI-AV dataset. Used for closed-loop simulation and "
            "evaluation with AlpaSim. ⚠️ USDZ format — not loadable in this app."
        ),
        "size": "1.77 TB",
        "url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec",
    },
    # ── Public driving datasets (used in Alpamayo 1.5 training) ───────────
    "coda-lm": {
        "hf_id": "KaiChen1998/coda-lm-llava-format",
        "loader": "hf_streaming",
        "description": "Corner case analysis with multi-turn conversations on driving images.",
        "size": "~1 GB",
        "url": "https://huggingface.co/datasets/DLCV-BUAA/CODA-LM",
        "image_keys": ["image"],
        "qa_mode": "conversations",
    },
    "drive-action": {
        "hf_id": "LiAuto-DriveAction/drive-action",
        "loader": "hf_streaming",
        "description": "Bilingual (EN/CN) multiple-choice VQA with multi-view driving images.",
        "size": "~2 GB",
        "url": "https://huggingface.co/datasets/LiAuto-DriveAction/drive-action",
        "image_keys": ["image_0", "image_1", "image_2"],
        "qa_mode": "drive_action",
    },
    "lingoqa": {
        "hf_id": "runoob1/lingoqa",
        "loader": "hf_streaming",
        "description": "Language-grounded driving QA with 5 sequential video frames per sample.",
        "size": "~1 GB",
        "url": "https://huggingface.co/datasets/wayveai/LingoQA",
        "image_keys": ["image_1", "image_2", "image_3", "image_4", "image_5"],
        "qa_mode": "question_answer",
    },
    "nuscenesqa": {
        "hf_id": "KevinNotSmile/nuscenes-qa-mini",
        "loader": "hf_streaming",
        "description": "VQA on nuScenes with 6 camera views + LiDAR BEV.",
        "size": "~500 MB",
        "url": "https://huggingface.co/datasets/qiantianwen/NuScenes-QA",
        "image_keys": ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
                        "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"],
        "qa_mode": "question_answer",
    },
    "drivegpt4": {
        "hf_id": "owl10/Drivegpt4-BDD",
        "loader": "hf_streaming",
        "description": "Multi-modal driving dialogue with BDD video frames.",
        "size": "~2 GB",
        "url": "https://huggingface.co/datasets/OpenDriveLab/DriveGPT4",
        "image_keys": ["png"],
        "qa_mode": "metadata_only",
    },
    # ── Datasets with image paths (not embedded) — stream metadata only ──
    "drivelm": {
        "hf_id": "OpenDriveLab/DriveLM",
        "loader": "hf_streaming",
        "description": "Driving language-model QA pairs with nuScenes multi-view images.",
        "size": "~2 GB (gated — requires access approval)",
        "url": "https://huggingface.co/datasets/OpenDriveLab/DriveLM",
        "image_keys": [],
        "qa_mode": "metadata_only",
    },
    "omnidrive": {
        "hf_id": "runoob1/OmniDrive_test",
        "loader": "hf_streaming",
        "description": "Multi-task 3D driving VQA — scene, traffic, grounding, planning.",
        "size": "~1 GB",
        "url": "https://huggingface.co/datasets/NVlabs/OmniDrive",
        "image_keys": [],
        "qa_mode": "question_answer",
    },
    "nuinstruct": {
        "hf_id": "runoob1/NuInstruct_test_accuracytask",
        "loader": "hf_streaming",
        "description": "Instruction-following VQA on nuScenes with multiple sub-tasks.",
        "size": "~500 MB",
        "url": "https://huggingface.co/datasets/nuInstruct/nuInstruct",
        "image_keys": [],
        "qa_mode": "question_answer",
    },
    # ── Non-standard / special format datasets ───────────────────────────
    "navsim": {
        "hf_id": "autonomousvision/navsim",
        "loader": "unsupported",
        "description": (
            "Navigation simulation benchmark — custom OpenScene format. "
            "⚠️ Not loadable via standard HF streaming."
        ),
        "size": "variable",
        "url": "https://huggingface.co/datasets/autonomousvision/navsim",
    },
    "talk2car": {
        "hf_id": "talk2car/Talk2Car",
        "loader": "unsupported",
        "description": (
            "Natural language grounding in driving scenes — JSON + nuScenes images. "
            "⚠️ Requires nuScenes image download separately."
        ),
        "size": "~500 MB",
        "url": "https://huggingface.co/datasets/talk2car/Talk2Car",
    },
    "maplm": {
        "hf_id": "LLVM-AD/maplm_v2",
        "loader": "unsupported",
        "description": (
            "Map-centric driving VQA — custom loader script, not viewer-compatible. "
            "⚠️ Requires custom loading."
        ),
        "size": "variable",
        "url": "https://huggingface.co/datasets/MapLM/MapLM",
    },
    "mm-au": {
        "hf_id": "Yiming-Li/MM-AU",
        "loader": "unsupported",
        "description": (
            "Multi-modal action understanding — raw zip files only. "
            "⚠️ No structured schema available."
        ),
        "size": "variable",
        "url": "https://huggingface.co/datasets/Yiming-Li/MM-AU",
    },
    "senna": {
        "hf_id": "hustvl/Senna",
        "loader": "unsupported",
        "description": (
            "Senna is a model/system, not a downloadable dataset. "
            "⚠️ Generates VQA data from nuScenes using LLaVA."
        ),
        "size": "N/A",
        "url": "https://huggingface.co/hustvl/Senna",
    },
    "roadwork": {
        "hf_id": "anuragxel/roadwork-dataset",
        "loader": "unsupported",
        "description": (
            "Construction zone data — raw zip files (images, video, segmentation). "
            "⚠️ No structured schema on HF."
        ),
        "size": "variable",
        "url": "https://huggingface.co/datasets/anuragxel/roadwork-dataset",
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
    if loader == "unsupported":
        raise ValueError(
            f"Dataset '{dataset_key}' is not loadable in this app.\n"
            f"{ds_info['description']}"
        )
    if loader == "physical_ai_av_sdk":
        return _load_physical_ai_av_sdk(config, num_samples)
    if loader == "hf_parquet":
        return _load_hf_parquet(
            ds_info["hf_id"], ds_info.get("parquet_path"), num_samples,
        )
    # hf_streaming — normalize samples after loading
    raw_samples = _load_hf_streaming(ds_info["hf_id"], num_samples)
    return [_normalize_sample(s, ds_info) for s in raw_samples]


# ── Sample normalization ──────────────────────────────────────────────────────
# All HF streaming datasets are normalized to a common schema:
#   images:   list[PIL.Image] — extracted from known image keys
#   question: str             — extracted from QA fields
#   answer:   str             — extracted from QA fields
#   metadata: dict            — all remaining fields
#   source:   str             — "hf_streaming"

def _normalize_sample(raw: dict, ds_info: dict) -> dict:
    """Normalize a raw HF dataset row to the common sample schema."""
    sample = {"source": "hf_streaming", "metadata": {}}

    image_keys = ds_info.get("image_keys", [])
    qa_mode = ds_info.get("qa_mode", "metadata_only")

    # ── Extract images ────────────────────────────────────────────────
    images = []
    used_keys = set()

    # Try configured image keys first
    for key in image_keys:
        val = raw.get(key)
        if val is not None:
            img = _to_pil_image(val)
            if img is not None:
                images.append(img)
                used_keys.add(key)

    # Auto-detect image fields if none configured
    if not images:
        for key, val in raw.items():
            if _is_image_value(val):
                img = _to_pil_image(val)
                if img is not None:
                    images.append(img)
                    used_keys.add(key)

    if images:
        sample["images"] = images

    # ── Extract QA text ───────────────────────────────────────────────
    if qa_mode == "conversations":
        # CODA-LM style: conversations = [{from: "human", value: "..."}, {from: "gpt", value: "..."}]
        convos = raw.get("conversations", [])
        if isinstance(convos, (list, tuple)) and convos:
            q_parts, a_parts = [], []
            for turn in convos:
                if isinstance(turn, dict):
                    role = turn.get("from", "")
                    text = turn.get("value", "")
                    if role in ("human", "user"):
                        q_parts.append(text)
                    elif role in ("gpt", "assistant"):
                        a_parts.append(text)
            sample["question"] = "\n".join(q_parts) if q_parts else ""
            sample["answer"] = "\n".join(a_parts) if a_parts else ""
        used_keys.add("conversations")

    elif qa_mode == "drive_action":
        # Drive-Action style: content_en = {question, answer, options: {A, B, C, D}}
        content = raw.get("content_en") or raw.get("content_cn") or {}
        if isinstance(content, dict):
            q = content.get("question", "")
            opts = content.get("options", {})
            if isinstance(opts, dict):
                opt_text = "\n".join(f"  {k}. {v}" for k, v in sorted(opts.items()))
                q = f"{q}\n{opt_text}" if opt_text else q
            sample["question"] = q
            sample["answer"] = content.get("answer", "")
        else:
            sample["question"] = raw.get("question", "")
            sample["answer"] = raw.get("answer", "")
        used_keys.update({"content_en", "content_cn", "question_category",
                          "qa_l0", "qa_l1", "question_slice_id"})

    elif qa_mode == "question_answer":
        sample["question"] = str(raw.get("question", ""))
        sample["answer"] = str(raw.get("answer", ""))
        used_keys.update({"question", "answer"})

    else:
        # metadata_only — try to detect QA fields generically
        for qk in ("question", "query", "prompt", "instruction", "input"):
            if qk in raw:
                sample["question"] = str(raw[qk])
                used_keys.add(qk)
                break
        for ak in ("answer", "response", "output", "text", "completion"):
            if ak in raw:
                sample["answer"] = str(raw[ak])
                used_keys.add(ak)
                break

    # ── Remaining fields → metadata ──────────────────────────────────
    for key, val in raw.items():
        if key not in used_keys:
            # Skip large binary/image data in metadata
            if _is_image_value(val):
                sample["metadata"][key] = f"<image: {type(val).__name__}>"
            elif isinstance(val, (bytes, bytearray)):
                sample["metadata"][key] = f"<bytes: {len(val)}>"
            else:
                sample["metadata"][key] = val

    return sample


def _to_pil_image(val):
    """Try to convert a value to a PIL Image. Returns None on failure."""
    try:
        from PIL import Image
        if isinstance(val, Image.Image):
            return val
    except ImportError:
        return None

    # numpy array
    if isinstance(val, np.ndarray) and val.ndim in (2, 3):
        try:
            from PIL import Image
            if val.dtype != np.uint8:
                if val.max() <= 1.0:
                    val = (val * 255).clip(0, 255).astype(np.uint8)
                else:
                    val = val.clip(0, 255).astype(np.uint8)
            return Image.fromarray(val)
        except Exception:
            return None

    # dict with "bytes" key (HF Image feature decoded)
    if isinstance(val, dict) and "bytes" in val:
        try:
            import io
            from PIL import Image
            return Image.open(io.BytesIO(val["bytes"]))
        except Exception:
            return None

    return None


def _is_image_value(val) -> bool:
    """Check if a value looks like image data."""
    try:
        from PIL import Image
        if isinstance(val, Image.Image):
            return True
    except ImportError:
        pass
    if isinstance(val, np.ndarray) and val.ndim in (2, 3) and val.shape[-1] in (1, 3, 4):
        return True
    if isinstance(val, dict) and "bytes" in val:
        return True
    return False


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
