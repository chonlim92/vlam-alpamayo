"""Dataset utilities for loading driving scene data."""

import random

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
    # ── Public driving datasets (images + VQA, loadable) ──────────────────
    "coda-lm": {
        "hf_id": "KaiChen1998/coda-lm-llava-format",
        "hf_config": "English",
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
        "hf_config": "day",
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
}

# Ordered list of dataset keys for dropdowns / CLI choices
DATASET_KEYS = list(DATASETS.keys())


def load_sample_data(
    config: AppConfig,
    dataset_key: str = "physical-ai-av",
    num_samples: int = 1,
    data_mode: str = "auto",
):
    """Load sample data from the selected dataset.

    Args:
        config: Application configuration (needs HF token).
        dataset_key: Key from DATASETS registry.
        num_samples: Number of samples to load.
        data_mode: Loading mode for physical-ai-av:
            - ``"video"`` — always use SDK to download camera video clips.
            - ``"parquet"`` — load text-only CoC annotations from parquet.
            - ``"auto"`` — use the dataset’s configured loader.

    Returns:
        List of data samples.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Choose from: {DATASET_KEYS}")

    ds_info = DATASETS[dataset_key]
    login(token=config.huggingface_token)

    loader = ds_info["loader"]

    # Override loader for physical-ai-av based on data_mode
    if dataset_key == "physical-ai-av" and data_mode != "auto":
        if data_mode == "video":
            loader = "physical_ai_av_sdk"
        elif data_mode == "parquet":
            loader = "hf_parquet"

    if loader == "unsupported":
        raise ValueError(
            f"Dataset '{dataset_key}' is not loadable in this app.\n"
            f"{ds_info['description']}"
        )
    if loader == "physical_ai_av_sdk":
        return _load_physical_ai_av_sdk(config, num_samples)
    if loader == "hf_parquet":
        parquet_path = ds_info.get("parquet_path", "reasoning/ood_reasoning.parquet")
        return _load_hf_parquet(
            ds_info["hf_id"], parquet_path, num_samples,
        )
    # hf_streaming — normalize samples after loading
    raw_samples = _load_hf_streaming(ds_info["hf_id"], num_samples, config=ds_info.get("hf_config"))
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

    # Pick candidate clip IDs — request extra to account for clips without video
    max_candidates = num_samples * 3
    clip_ids = _select_clip_ids(ds, reasoning_df, max_candidates)

    camera_feature = "camera_front_wide_120fov"
    samples = []
    skipped = 0

    for i, clip_id in enumerate(clip_ids):
        if len(samples) >= num_samples:
            break

        print(f"[{i + 1}/{len(clip_ids)}] Downloading clip {clip_id}...")

        sample = {"clip_id": clip_id, "source": "physical_ai_av_sdk"}

        # Download and extract camera frames
        has_video = False
        try:
            ds.download_clip_features(clip_id, features=[camera_feature])
            video_reader = ds.get_clip_feature(clip_id, camera_feature)
            frames = _extract_video_frames(video_reader, max_frames=40)
            if frames:
                sample["camera_front_wide_120fov"] = frames
                has_video = True
                print(f"  Camera: {len(frames)} frames extracted")
            else:
                print("  Camera: no frames extracted (reader returned empty)")
        except Exception as e:
            print(f"  Camera download failed: {e}")

        # Skip clips without video data
        if not has_video:
            skipped += 1
            print(f"  ⏭️  Skipping clip (no video) — {skipped} skipped so far")
            continue

        # Download egomotion and extract trajectory
        try:
            ds.download_clip_features(clip_id, features=["egomotion"])
            ego = ds.get_clip_feature(clip_id, "egomotion")
            sample["egomotion"] = ego
            # Extract trajectory waypoints from egomotion
            ego_xyz = _extract_ego_xyz_from_egomotion(ego)
            if ego_xyz is not None:
                sample["ego_history_xyz"] = ego_xyz
                sample["trajectory"] = ego_xyz[:, :2]  # x,y for BEV plot
                print(f"  Egomotion: loaded ({len(ego_xyz)} waypoints, shape {ego_xyz.shape})")
                # Also extract rotation (needed by model)
                ego_rot = _extract_ego_rot_from_egomotion(ego)
                if ego_rot is not None:
                    sample["ego_history_rot"] = ego_rot
                    print(f"  Rotation: {ego_rot.shape}")
            else:
                print("  Egomotion: loaded (no xyz extracted — see diagnostics)")
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
        raise RuntimeError(
            f"No clips with video data could be loaded (tried {len(clip_ids)}, "
            f"skipped {skipped}). Check dataset access and network."
        )

    if skipped:
        print(f"✅ Loaded {len(samples)} clip(s) with video ({skipped} without video skipped)")

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
    """Select clip IDs to download — only clips that have camera data."""
    camera_feature = "camera_front_wide_120fov"
    clip_idx = ds.clip_index

    # Log clip_index columns to help debug feature availability
    print(f"  Clip index: {len(clip_idx)} clips, columns: {list(clip_idx.columns)}")

    # Filter clip_index for clips with camera data
    camera_clips = _filter_clips_with_feature(clip_idx, camera_feature)

    if reasoning_df is not None and len(reasoning_df) > 0:
        # Intersect OOD-annotated clips with camera-available clips
        annotated_ids = set(reasoning_df.index)
        if camera_clips is not None:
            camera_ids = set(camera_clips.index)
            both = list(annotated_ids & camera_ids)
            if both:
                random.shuffle(both)
                ids = both[:num_samples]
                print(f"Selected {len(ids)} OOD-annotated clip(s) with camera data "
                      f"(from {len(both)} candidates)")
                return ids
            # No overlap — try camera-only clips
            print(f"  No OOD-annotated clips have camera data. "
                  f"Using camera-available clips instead.")
            pool = camera_clips.index.tolist()
            random.shuffle(pool)
            ids = pool[:num_samples]
            if ids:
                print(f"Selected {len(ids)} camera-available clip(s)")
                return ids
        else:
            # Can't filter by camera — use annotated clips and hope for the best
            pool = reasoning_df.index.tolist()
            random.shuffle(pool)
            ids = pool[:num_samples]
            if ids:
                print(f"Selected {len(ids)} OOD-annotated clip(s) (camera availability unknown)")
                return ids

    # No reasoning annotations — use camera-filtered clips if possible
    if camera_clips is not None and len(camera_clips) > 0:
        pool = camera_clips.index.tolist()
        random.shuffle(pool)
        ids = pool[:num_samples]
        print(f"Selected {len(ids)} camera-available clip(s) from clip index")
        return ids

    # Final fallback — unfiltered clip index
    pool = clip_idx.index.tolist()
    random.shuffle(pool)
    ids = pool[:num_samples]
    print(f"Selected {len(ids)} clip(s) from clip index (camera availability unknown)")
    return ids


def _filter_clips_with_feature(clip_idx, feature_name: str):
    """Filter clip_index DataFrame for clips that have a specific feature.

    The clip_index may have:
    - A column named after the feature (boolean or presence marker)
    - A 'features' column listing available features per clip
    - Feature columns prefixed with 'has_' or suffixed with '_available'
    Returns filtered DataFrame, or None if we can't determine availability.
    """
    cols = list(clip_idx.columns)
    col_lower = {c.lower(): c for c in cols}

    # Direct column match (e.g. "camera_front_wide_120fov")
    if feature_name in cols:
        col = feature_name
        try:
            # Boolean column or non-null check
            if clip_idx[col].dtype == bool:
                return clip_idx[clip_idx[col]]
            return clip_idx[clip_idx[col].notna()]
        except Exception:
            pass

    # has_<feature> or <feature>_available
    for prefix_suffix in (f"has_{feature_name}", f"{feature_name}_available"):
        if prefix_suffix.lower() in col_lower:
            col = col_lower[prefix_suffix.lower()]
            try:
                return clip_idx[clip_idx[col].astype(bool)]
            except Exception:
                pass

    # 'features' list column — check if feature_name appears in each row's list
    if "features" in col_lower:
        col = col_lower["features"]
        try:
            mask = clip_idx[col].apply(
                lambda x: feature_name in x if isinstance(x, (list, set, tuple)) else False
            )
            filtered = clip_idx[mask]
            if len(filtered) > 0:
                return filtered
        except Exception:
            pass

    # Check for any column containing "camera" and "front"
    for c in cols:
        cl = c.lower()
        if "camera" in cl and "front" in cl:
            try:
                if clip_idx[c].dtype == bool:
                    filtered = clip_idx[clip_idx[c]]
                else:
                    filtered = clip_idx[clip_idx[c].notna()]
                if len(filtered) > 0:
                    print(f"  Filtered by column '{c}': {len(filtered)} clips with camera data")
                    return filtered
            except Exception:
                continue

    print(f"  Could not determine camera availability from clip_index columns: {cols}")
    return None


def _extract_video_frames(video_reader, max_frames: int = 40) -> list[np.ndarray]:
    """Extract numpy BGR frames from a SeekVideoReader or similar object.

    SeekVideoReader API (physical_ai_av SDK):
      - timestamps: np.ndarray of PTS values
      - decode_images_from_frame_indices(indices) -> list of PIL Images
      - decode_images_from_timestamps(timestamps) -> list of PIL Images
      - fps, container, stream, time_base, close()
    """
    frames = []

    # ── Strategy 1: decode_images_from_frame_indices (preferred) ─────
    try:
        if hasattr(video_reader, 'decode_images_from_frame_indices'):
            ts = video_reader.timestamps
            n_total = len(ts)
            step = max(1, n_total // max_frames)
            indices = np.array(range(0, n_total, step))[:max_frames]
            images = video_reader.decode_images_from_frame_indices(indices)
            for img in images:
                arr = _frame_to_numpy(img)
                if arr is not None:
                    frames.append(arr)
            if frames:
                print(f"  Extracted {len(frames)} frames via decode_images_from_frame_indices")
                return frames
    except Exception as e:
        print(f"  decode_images_from_frame_indices failed: {e}")

    # ── Strategy 2: decode_images_from_timestamps ────────────────────
    try:
        if hasattr(video_reader, 'decode_images_from_timestamps'):
            ts = video_reader.timestamps
            n_total = len(ts)
            step = max(1, n_total // max_frames)
            selected_ts = ts[::step][:max_frames]
            images = video_reader.decode_images_from_timestamps(selected_ts)
            for img in images:
                arr = _frame_to_numpy(img)
                if arr is not None:
                    frames.append(arr)
            if frames:
                print(f"  Extracted {len(frames)} frames via decode_images_from_timestamps")
                return frames
    except Exception as e:
        print(f"  decode_images_from_timestamps failed: {e}")

    # ── Strategy 3: Decode directly from the underlying av container ─
    try:
        if hasattr(video_reader, 'container'):
            container = video_reader.container
            for av_frame in container.decode(video=0):
                arr = av_frame.to_ndarray(format="bgr24")
                frames.append(arr)
                if len(frames) >= max_frames:
                    break
            if frames:
                print(f"  Extracted {len(frames)} frames via container.decode()")
                return frames
    except Exception as e:
        print(f"  container.decode() failed: {e}")

    vr_type = type(video_reader).__name__
    print(f"  ❌ All extraction strategies failed for {vr_type}")
    return frames



def _extract_ego_xyz_from_egomotion(ego) -> np.ndarray | None:
    """Extract x,y,z ego trajectory from an egomotion object.

    Returns an (N, 3) float64 array of [x, y, z] positions, or None.
    The model expects this as ``sample["ego_history_xyz"]``.

    Egomotion may be a DataFrame, dict of arrays, structured array,
    or custom SDK object with various column naming conventions.
    """
    try:
        ego_type = type(ego).__name__
        print(f"  Egomotion type: {ego_type}")

        df = None
        if isinstance(ego, pd.DataFrame):
            df = ego
        elif hasattr(ego, 'to_pandas'):
            df = ego.to_pandas()
        elif hasattr(ego, 'columns'):
            df = pd.DataFrame(ego)
        elif isinstance(ego, dict):
            print(f"  Egomotion keys: {list(ego.keys())[:20]}")
            df = pd.DataFrame(ego)
        elif isinstance(ego, np.ndarray):
            print(f"  Egomotion shape: {ego.shape}, dtype: {ego.dtype}")
            if ego.ndim == 2:
                if ego.shape[1] >= 3:
                    return ego[:, :3].astype(np.float64)
                elif ego.shape[1] == 2:
                    z = np.zeros((ego.shape[0], 1))
                    return np.hstack([ego[:, :2], z]).astype(np.float64)
            return None
        else:
            attrs = [a for a in dir(ego) if not a.startswith('_')]
            print(f"  Egomotion attrs: {attrs[:20]}")

            # Handle Interpolator objects (physical_ai_av SDK)
            if hasattr(ego, 'values'):
                values = ego.values
                vtype = getattr(ego, 'value_type', None)
                print(f"  Egomotion value_type: {vtype}, values type: {type(values).__name__}")

                # If values is a single EgomotionState (not array/list),
                # inspect it and try to call the Interpolator at its timestamps
                if not isinstance(values, (np.ndarray, list, tuple)):
                    val_attrs = [a for a in dir(values) if not a.startswith('_')]
                    print(f"  EgomotionState attrs: {val_attrs}")
                    # Log any array-like attributes on the state object
                    for attr in val_attrs:
                        try:
                            v = getattr(values, attr)
                            if isinstance(v, np.ndarray):
                                print(f"    .{attr}: ndarray shape={v.shape} dtype={v.dtype}")
                            elif isinstance(v, (list, tuple)) and len(v) > 0:
                                print(f"    .{attr}: {type(v).__name__} len={len(v)}, [0]={type(v[0]).__name__}")
                            else:
                                vstr = str(v)
                                if len(vstr) > 120:
                                    vstr = vstr[:120] + "..."
                                print(f"    .{attr}: {type(v).__name__} = {vstr}")
                        except Exception as e:
                            print(f"    .{attr}: <error: {e}>")

                    # Try to evaluate Interpolator at its own timestamps
                    if hasattr(ego, 'timestamps'):
                        ts = ego.timestamps
                        print(f"  Interpolator timestamps: type={type(ts).__name__}, len={len(ts) if hasattr(ts, '__len__') else '?'}")
                        # Try calling interpolator
                        for call_method in ['__call__', 'interpolate', 'evaluate', 'sample']:
                            if hasattr(ego, call_method) or call_method == '__call__':
                                try:
                                    if call_method == '__call__':
                                        states = ego(ts)
                                    else:
                                        states = getattr(ego, call_method)(ts)
                                    print(f"  ego({call_method}) returned: {type(states).__name__}")
                                    result = _extract_xyz_from_states(states)
                                    if result is not None:
                                        return result
                                    break
                                except Exception as e:
                                    print(f"  ego.{call_method}(timestamps) failed: {e}")

                    # Try extracting from the single EgomotionState directly
                    # (it may contain the full trajectory as array attributes)
                    for pos_attr in ('position', 'translation', 'xyz', 'pose',
                                     'position_m', 'translation_m'):
                        if hasattr(values, pos_attr):
                            pos = getattr(values, pos_attr)
                            print(f"  EgomotionState.{pos_attr}: type={type(pos).__name__}")
                            if isinstance(pos, np.ndarray):
                                print(f"    shape={pos.shape}")
                                if pos.ndim == 2 and pos.shape[1] >= 3:
                                    return pos[:, :3].astype(np.float64)
                                elif pos.ndim == 2 and pos.shape[1] == 2:
                                    z = np.zeros((pos.shape[0], 1))
                                    return np.hstack([pos, z]).astype(np.float64)
                                elif pos.ndim == 1 and len(pos) >= 3:
                                    return pos[:3].reshape(1, 3).astype(np.float64)

                    return None

                if isinstance(values, np.ndarray):
                    print(f"  Egomotion values shape: {values.shape}, dtype: {values.dtype}")
                    if values.ndim == 2 and values.shape[1] >= 3:
                        return values[:, :3].astype(np.float64)
                    elif values.ndim == 2 and values.shape[1] == 2:
                        z = np.zeros((values.shape[0], 1))
                        return np.hstack([values, z]).astype(np.float64)
                    elif values.ndim == 1:
                        # Might be structured array or flat — log for debugging
                        print(f"  Egomotion values (1D): first 3 = {values[:3]}")
                elif isinstance(values, (list, tuple)) and len(values) > 0:
                    # List of poses/transforms — try to extract translation
                    first = values[0]
                    print(f"  Egomotion values[0] type: {type(first).__name__}")
                    if isinstance(first, np.ndarray):
                        arr = np.array(values)
                        print(f"  Egomotion values array shape: {arr.shape}")
                        if arr.ndim == 2 and arr.shape[1] >= 3:
                            return arr[:, :3].astype(np.float64)
                        elif arr.ndim == 3 and arr.shape[1] == 4 and arr.shape[2] == 4:
                            # 4x4 transformation matrices — translation is [:3, 3]
                            translations = arr[:, :3, 3].astype(np.float64)
                            print(f"  Extracted translations from 4x4 matrices: {translations.shape}")
                            return translations
                    elif hasattr(first, 'translation'):
                        # Pose objects with .translation attribute
                        xyz = []
                        for v in values:
                            t = v.translation
                            if isinstance(t, np.ndarray):
                                xyz.append(t[:3])
                            elif hasattr(t, 'x'):
                                xyz.append([t.x, t.y, getattr(t, 'z', 0)])
                        if xyz:
                            return np.array(xyz, dtype=np.float64)
                    elif isinstance(first, (list, tuple)):
                        arr = np.array(values, dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] >= 3:
                            return arr[:, :3]

            # Try accessing as attribute-based object
            for x_attr, y_attr, z_attr in [
                ('x', 'y', 'z'), ('position_x', 'position_y', 'position_z'),
                ('tx', 'ty', 'tz'),
            ]:
                if hasattr(ego, x_attr) and hasattr(ego, y_attr):
                    x = np.array(getattr(ego, x_attr))
                    y = np.array(getattr(ego, y_attr))
                    z = np.array(getattr(ego, z_attr)) if hasattr(ego, z_attr) else np.zeros_like(x)
                    return np.stack([x, y, z], axis=1).astype(np.float64)
            return None

        if df is not None:
            print(f"  Egomotion columns: {list(df.columns)}")
            print(f"  Egomotion rows: {len(df)}")

            # Try common xyz column naming patterns
            for x_col, y_col, z_col in [
                ('x', 'y', 'z'),
                ('position_x', 'position_y', 'position_z'),
                ('pos_x', 'pos_y', 'pos_z'),
                ('tx', 'ty', 'tz'),
                ('translation_x', 'translation_y', 'translation_z'),
            ]:
                if x_col in df.columns and y_col in df.columns:
                    x = df[x_col].values.astype(np.float64)
                    y = df[y_col].values.astype(np.float64)
                    z = df[z_col].values.astype(np.float64) if z_col in df.columns else np.zeros_like(x)
                    print(f"  Using columns '{x_col}', '{y_col}', '{z_col}' for ego_history_xyz")
                    return np.stack([x, y, z], axis=1)

            # Try columns containing position hints
            cols_lower = {c.lower(): c for c in df.columns}
            x_cands = [c for c in cols_lower if 'x' in c and ('pos' in c or 'trans' in c)]
            y_cands = [c for c in cols_lower if 'y' in c and ('pos' in c or 'trans' in c)]
            z_cands = [c for c in cols_lower if 'z' in c and ('pos' in c or 'trans' in c)]
            if x_cands and y_cands:
                xc = cols_lower[x_cands[0]]
                yc = cols_lower[y_cands[0]]
                x = df[xc].values.astype(np.float64)
                y = df[yc].values.astype(np.float64)
                if z_cands:
                    zc = cols_lower[z_cands[0]]
                    z = df[zc].values.astype(np.float64)
                else:
                    z = np.zeros_like(x)
                print(f"  Using columns '{xc}', '{yc}' for ego_history_xyz")
                return np.stack([x, y, z], axis=1)

            # Fallback: first 2-3 numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                print(f"  Fallback: using numeric columns {numeric_cols[:3]} for ego_history_xyz")
                return df[numeric_cols[:3]].values.astype(np.float64)
            elif len(numeric_cols) >= 2:
                print(f"  Fallback: using numeric columns {numeric_cols[:2]} + zeros for z")
                xy = df[numeric_cols[:2]].values.astype(np.float64)
                z = np.zeros((len(xy), 1))
                return np.hstack([xy, z])

    except Exception as e:
        print(f"  ego_history_xyz extraction error: {e}")

    return None


def _extract_xyz_from_rigid_transform(pose) -> np.ndarray | None:
    """Extract xyz translation from a RigidTransform object.

    RigidTransform may expose translation via:
    - .translation property (ndarray)
    - .as_matrix() / .matrix / .to_matrix() → 4x4 or Nx4x4 matrices
    - .position property
    """
    pose_attrs = [a for a in dir(pose) if not a.startswith('_')]
    print(f"  RigidTransform attrs: {pose_attrs}")

    # Try .translation directly
    if hasattr(pose, 'translation'):
        t = pose.translation
        print(f"  RigidTransform.translation: type={type(t).__name__}")
        if isinstance(t, np.ndarray):
            print(f"    shape={t.shape}, dtype={t.dtype}")
            if t.ndim == 2 and t.shape[1] >= 3:
                return t[:, :3].astype(np.float64)
            elif t.ndim == 1 and len(t) >= 3:
                return t[:3].reshape(1, 3).astype(np.float64)

    # Try getting the 4x4 matrix(ces)
    mat = _get_rigid_transform_matrices(pose)
    if mat is not None:
        if mat.ndim == 3 and mat.shape[1] == 4 and mat.shape[2] == 4:
            return mat[:, :3, 3].astype(np.float64)
        elif mat.ndim == 2 and mat.shape == (4, 4):
            return mat[:3, 3].reshape(1, 3).astype(np.float64)

    # Try .position
    if hasattr(pose, 'position'):
        p = pose.position
        if isinstance(p, np.ndarray):
            print(f"  RigidTransform.position: shape={p.shape}")
            if p.ndim == 2 and p.shape[1] >= 3:
                return p[:, :3].astype(np.float64)

    # Log repr for debugging
    pose_repr = repr(pose)
    if len(pose_repr) > 200:
        pose_repr = pose_repr[:200] + "..."
    print(f"  RigidTransform repr: {pose_repr}")

    return None


def _get_rigid_transform_matrices(pose) -> np.ndarray | None:
    """Get the raw 4x4 matrices from a RigidTransform object."""
    for matrix_method in ('as_matrix', 'matrix', 'to_matrix', 'numpy',
                          'as_numpy', 'homogeneous'):
        if hasattr(pose, matrix_method):
            try:
                val = getattr(pose, matrix_method)
                mat = val() if callable(val) else val
                if isinstance(mat, np.ndarray):
                    print(f"  RigidTransform.{matrix_method}: shape={mat.shape}")
                    if mat.ndim in (2, 3):
                        return mat
            except Exception as e:
                print(f"  RigidTransform.{matrix_method} failed: {e}")
    return None


def _extract_rot_from_rigid_transform(pose) -> np.ndarray | None:
    """Extract rotation quaternions (N, 4) [w,x,y,z] from a RigidTransform."""
    # Try .rotation property first
    if hasattr(pose, 'rotation'):
        rot = pose.rotation
        print(f"  RigidTransform.rotation: type={type(rot).__name__}")
        if isinstance(rot, np.ndarray):
            print(f"    shape={rot.shape}")
            if rot.ndim == 2:
                if rot.shape[1] == 4:  # already quaternions
                    return rot.astype(np.float64)
                if rot.shape[1] == 9:  # flattened 3x3
                    rot = rot.reshape(-1, 3, 3)
            if rot.ndim == 3 and rot.shape[1:] == (3, 3):
                return _rotmat_to_quat(rot)
        # Might be a Rotation-like object
        if hasattr(rot, 'as_quat'):
            q = rot.as_quat()  # scipy returns [x,y,z,w]
            if isinstance(q, np.ndarray):
                if q.ndim == 2 and q.shape[1] == 4:
                    return q[:, [3, 0, 1, 2]].astype(np.float64)  # → [w,x,y,z]
                elif q.ndim == 1 and len(q) == 4:
                    return q[[3, 0, 1, 2]].reshape(1, 4).astype(np.float64)
        if hasattr(rot, 'as_matrix'):
            m = rot.as_matrix()
            if isinstance(m, np.ndarray):
                if m.ndim == 3 and m.shape[1:] == (3, 3):
                    return _rotmat_to_quat(m)

    # Fallback: extract from 4x4 matrices
    mat = _get_rigid_transform_matrices(pose)
    if mat is not None:
        if mat.ndim == 3 and mat.shape[1] == 4 and mat.shape[2] == 4:
            return _rotmat_to_quat(mat[:, :3, :3])
        elif mat.ndim == 2 and mat.shape == (4, 4):
            return _rotmat_to_quat(mat[:3, :3].reshape(1, 3, 3))

    return None


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert (N, 3, 3) rotation matrices to (N, 4) quaternions [w,x,y,z]."""
    try:
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
        if q.ndim == 1:
            q = q.reshape(1, 4)
        return q[:, [3, 0, 1, 2]].astype(np.float64)  # → [w,x,y,z]
    except ImportError:
        pass

    # Manual Shepperd's method fallback
    N = R.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        m = R[i]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            q[i] = [0.25 / s, (m[2, 1] - m[1, 2]) * s,
                    (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                    (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                    0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return q


def _extract_ego_rot_from_egomotion(ego) -> np.ndarray | None:
    """Extract rotation quaternions (N, 4) [w,x,y,z] from an egomotion object."""
    try:
        # Direct path: Interpolator → EgomotionState → pose → RigidTransform
        if hasattr(ego, 'values') and hasattr(ego.values, 'pose'):
            return _extract_rot_from_rigid_transform(ego.values.pose)

        # Try calling the Interpolator and getting pose from result
        if hasattr(ego, 'timestamps') and callable(ego):
            states = ego(ego.timestamps)
            if hasattr(states, 'pose'):
                return _extract_rot_from_rigid_transform(states.pose)
    except Exception as e:
        print(f"  ego_history_rot extraction error: {e}")
    return None


def _extract_xyz_from_states(states) -> np.ndarray | None:
    """Extract xyz positions from a collection of EgomotionState objects."""
    if isinstance(states, np.ndarray):
        print(f"  States array: shape={states.shape}, dtype={states.dtype}")
        if states.ndim == 2 and states.shape[1] >= 3:
            return states[:, :3].astype(np.float64)
        if states.ndim == 3 and states.shape[1] == 4 and states.shape[2] == 4:
            return states[:, :3, 3].astype(np.float64)
        return None

    # Single state object — inspect it
    if not isinstance(states, (list, tuple)):
        s_attrs = [a for a in dir(states) if not a.startswith('_')]
        print(f"  States type: {type(states).__name__}, attrs: {s_attrs}")

        # Check for pose → RigidTransform → extract translation
        if hasattr(states, 'pose'):
            pose = states.pose
            print(f"  States.pose type: {type(pose).__name__}")
            xyz = _extract_xyz_from_rigid_transform(pose)
            if xyz is not None:
                return xyz

        # Check for position-like array attributes
        for attr in ('position', 'translation', 'xyz', 'position_m',
                     'translation_m'):
            if hasattr(states, attr):
                val = getattr(states, attr)
                if isinstance(val, np.ndarray):
                    print(f"  States.{attr}: shape={val.shape}")
                    if val.ndim == 2 and val.shape[1] >= 3:
                        return val[:, :3].astype(np.float64)
                    if val.ndim == 1 and len(val) >= 3:
                        return val[:3].reshape(1, 3).astype(np.float64)
        return None

    # List of states
    if len(states) == 0:
        return None

    first = states[0]
    print(f"  States list len={len(states)}, [0] type={type(first).__name__}")
    s_attrs = [a for a in dir(first) if not a.startswith('_')]
    print(f"  States[0] attrs: {s_attrs}")

    # Try common position attribute names
    for attr in ('pose',):
        if hasattr(first, attr):
            pose0 = getattr(first, attr)
            print(f"  States[0].{attr}: type={type(pose0).__name__}")
            xyz = _extract_xyz_from_rigid_transform(pose0)
            if xyz is not None:
                return xyz

    for attr in ('position', 'translation', 'xyz', 'position_m',
                 'translation_m'):
        if hasattr(first, attr):
            pos0 = getattr(first, attr)
            print(f"  States[0].{attr}: type={type(pos0).__name__}, value={pos0}")
            try:
                xyz = []
                for s in states:
                    p = getattr(s, attr)
                    if isinstance(p, np.ndarray):
                        xyz.append(p[:3])
                    elif hasattr(p, 'x'):
                        xyz.append([p.x, p.y, getattr(p, 'z', 0)])
                    elif isinstance(p, (list, tuple)):
                        xyz.append(list(p[:3]))
                    else:
                        break
                if len(xyz) == len(states):
                    arr = np.array(xyz, dtype=np.float64)
                    print(f"  Extracted xyz from .{attr}: shape={arr.shape}")
                    return arr
            except Exception as e:
                print(f"  .{attr} extraction failed: {e}")

    # Try converting list to numpy array directly
    try:
        arr = np.array(states)
        print(f"  np.array(states): shape={arr.shape}, dtype={arr.dtype}")
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float64)
    except Exception:
        pass

    return None


def _frame_to_numpy(frame) -> np.ndarray | None:
    """Convert various frame types to an RGB numpy array."""
    if frame is None:
        return None

    # Already numpy
    if isinstance(frame, np.ndarray):
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        if frame.ndim == 2:
            import cv2
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return frame

    # PIL Image — already RGB
    try:
        from PIL import Image
        if isinstance(frame, Image.Image):
            arr = np.array(frame.convert('RGB'))
            return arr
    except ImportError:
        pass

    # PyAV VideoFrame
    try:
        if hasattr(frame, "to_ndarray"):
            return frame.to_ndarray(format="rgb24")
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


def _load_hf_streaming(repo_id: str, num_samples: int, config: str | None = None) -> list:
    """Load from any HF dataset using streaming mode (no full download)."""
    try:
        kwargs = {"streaming": True, "trust_remote_code": True}
        if config:
            kwargs["name"] = config
        ds = load_dataset(repo_id, split="train", **kwargs)
    except Exception:
        # Some datasets only have a "test" or default split
        kwargs = {"streaming": True, "trust_remote_code": True}
        if config:
            kwargs["name"] = config
        ds = load_dataset(repo_id, **kwargs)
        # Take the first available split
        if hasattr(ds, "keys"):
            first_split = next(iter(ds.keys()))
            ds = ds[first_split]

    # Shuffle the stream so we get different samples each run
    ds = ds.shuffle(seed=random.randint(0, 2**31))
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
