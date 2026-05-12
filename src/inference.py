"""Inference engine for Alpamayo models."""

import json
from pathlib import Path
from datetime import datetime

import torch

from src.config import AppConfig
from src.model_loader import load_model, MODEL_INFO


class InferenceEngine:
    """Runs inference using Alpamayo models on driving scene data."""

    def __init__(self, config: AppConfig, model_key: str | None = None):
        self.config = config
        self.model_key = model_key or config.default_model
        self.model = None

    def load(self):
        """Load the model into memory."""
        self.model = load_model(self.config, self.model_key)
        return self

    def run_reasoning(self, data_sample) -> dict:
        """Run Chain-of-Causation reasoning on a data sample.

        Handles three types of data:
        1. Pre-annotated parquet rows → return CoC text directly
        2. Normalized HF streaming samples → show QA content + images
        3. SDK/model-ready samples with camera data → run full model inference

        Args:
            data_sample: A data sample dict.

        Returns:
            Dictionary with reasoning traces and trajectory predictions.
        """
        # ── Check if this is a pre-annotated parquet row ─────────────
        if self._is_parquet_row(data_sample):
            return self._format_parquet_result(data_sample)

        # ── Normalized HF streaming sample ───────────────────────────
        if data_sample.get("source") == "hf_streaming":
            return self._format_streaming_result(data_sample)

        # ── SDK sample with camera data ──────────────────────────────
        if data_sample.get("source") == "physical_ai_av_sdk":
            return self._format_sdk_result(data_sample)

        # ── Full model inference ─────────────────────────────────────
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        print("Running Chain-of-Causation reasoning...")

        result = self.model.sample_trajectories_from_data_with_vlm_rollout(
            data_sample,
            num_traj_samples=self.config.num_traj_samples,
        )

        output = {
            "model": MODEL_INFO[self.model_key]["name"],
            "reasoning_trace": result.get("reasoning", ""),
            "trajectory": self._format_trajectory(result.get("trajectory")),
            "timestamp": datetime.now().isoformat(),
        }

        return output

    @staticmethod
    def _is_parquet_row(sample) -> bool:
        """Check if the sample is a parquet metadata row (no camera data)."""
        if not isinstance(sample, dict):
            return False
        # Samples tagged with a source are handled by their own formatters
        if sample.get("source") in ("physical_ai_av_sdk", "hf_streaming"):
            return False
        keys = set(sample.keys())
        # ood_reasoning.parquet has: feature, event_cluster, events, split
        parquet_keys = {"event_cluster", "events", "feature", "split"}
        has_parquet_fields = len(parquet_keys & keys) >= 2
        # Model-ready samples would have camera/image data
        camera_keys = {"camera_front_wide_120fov", "images", "image", "video", "frames"}
        has_camera = bool(camera_keys & {k.lower() for k in keys})
        return has_parquet_fields and not has_camera

    def _format_parquet_result(self, sample: dict) -> dict:
        """Format a pre-annotated parquet row as an inference result.

        The ood_reasoning.parquet has columns:
          feature, event_cluster, events, split
        where `events` is a list of dicts: [{event_start_frame, event_start_timestamp, coc}, ...]
        """
        # Extract CoC reasoning from the nested events list
        events = sample.get("events", [])
        reasoning_parts = []
        if isinstance(events, (list, tuple)):
            for i, evt in enumerate(events, 1):
                if isinstance(evt, dict):
                    coc = evt.get("coc", "")
                    frame = evt.get("event_start_frame", "?")
                    ts = evt.get("event_start_timestamp", "?")
                    reasoning_parts.append(
                        f"#### Event {i} — Frame {frame}\n*Timestamp: {ts}*\n\n{coc}"
                    )
                elif isinstance(evt, str):
                    reasoning_parts.append(f"#### Event {i}\n{evt}")
        elif isinstance(events, str):
            # May be JSON string
            try:
                import json
                parsed = json.loads(events)
                if isinstance(parsed, list):
                    for i, evt in enumerate(parsed, 1):
                        coc = evt.get("coc", "") if isinstance(evt, dict) else str(evt)
                        frame = evt.get("event_start_frame", "?") if isinstance(evt, dict) else "?"
                        reasoning_parts.append(
                            f"#### Event {i} — Frame {frame}\n{coc}"
                        )
            except (json.JSONDecodeError, TypeError):
                reasoning_parts.append(events)

        reasoning_text = "\n\n".join(reasoning_parts) if reasoning_parts else "*(no events in this row)*"

        feature = sample.get("feature", "unknown")
        cluster = sample.get("event_cluster", "unknown")
        split = sample.get("split", "unknown")

        lines = [
            f"| Property | Value |",
            f"|---|---|",
            f"| **Feature** | {feature} |",
            f"| **Event Cluster** | {cluster} |",
            f"| **Split** | {split} |",
            f"| **Events** | {len(events) if isinstance(events, (list, tuple)) else '?'} |",
            "",
            "---",
            "### Chain-of-Causation (pre-annotated)\n",
            reasoning_text,
        ]

        return {
            "model": f"{MODEL_INFO[self.model_key]['name']} (pre-annotated data)",
            "reasoning_trace": "\n".join(lines),
            "trajectory": None,
            "timestamp": datetime.now().isoformat(),
            "source": "parquet_annotation",
        }

    def _format_streaming_result(self, sample: dict) -> dict:
        """Format a normalized HF streaming sample as a displayable result.

        Shows the QA content and metadata. Images are displayed via the
        visualization pipeline (the GUI checks for 'images' key).
        """
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        metadata = sample.get("metadata", {})
        images = sample.get("images", [])

        lines = []
        if images:
            lines.append(f"**Images:** {len(images)} frame(s)")
        if metadata.get("id"):
            lines.append(f"**ID:** `{metadata['id']}`")

        # Show QA content
        if question:
            lines.extend([
                "",
                "---",
                "### Question\n",
                question,
            ])
        if answer:
            lines.extend([
                "",
                "---",
                "### Ground-Truth Answer\n",
                answer,
            ])

        # Show non-trivial metadata
        meta_items = {k: v for k, v in metadata.items()
                      if k not in ("id",) and not str(v).startswith("<")}
        if meta_items:
            lines.extend([
                "",
                "---",
                "### Metadata\n",
            ])
            for k, v in meta_items.items():
                val_str = str(v)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "…"
                lines.append(f"- **{k}:** {val_str}")

        return {
            "model": f"{MODEL_INFO[self.model_key]['name']} (dataset sample)",
            "reasoning_trace": "\n".join(lines) if lines else "(no content in this sample)",
            "trajectory": None,
            "timestamp": datetime.now().isoformat(),
            "source": "hf_streaming",
            "has_images": bool(images),
        }

    def _format_sdk_result(self, sample: dict) -> dict:
        """Format a PhysicalAI-AV SDK sample.

        Shows camera frames as video + CoC reasoning from parquet annotations.
        Optionally runs model inference if loaded.
        """
        clip_id = sample.get("clip_id", "unknown")
        has_camera = "camera_front_wide_120fov" in sample
        has_ego = "egomotion" in sample

        # Try model inference if model is loaded and camera data is present
        trajectory = None
        model_reasoning = ""
        if self.model is not None and has_camera:
            try:
                print(f"Running model inference on clip {clip_id}...")
                model_result = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    sample,
                    num_traj_samples=self.config.num_traj_samples,
                )
                model_reasoning = model_result.get("reasoning", "")
                trajectory = self._format_trajectory(model_result.get("trajectory"))
            except Exception as e:
                model_reasoning = f"(model inference failed: {e})"
                print(f"  Model inference error: {e}")

        # Build reasoning text — combine model output + parquet CoC annotations
        n_frames = len(sample['camera_front_wide_120fov']) if has_camera else 0
        has_traj = "trajectory" in sample or "ego_history_xyz" in sample
        lines = [
            f"| Property | Value |",
            f"|---|---|",
            f"| **Clip ID** | `{clip_id}` |",
            f"| **Camera** | {n_frames} frames |" if has_camera else "| **Camera** | not available |",
            f"| **Egomotion** | {'loaded' if has_ego else 'not available'} |",
            f"| **Trajectory** | {'loaded' if has_traj else 'not available'} |",
        ]

        # Show trajectory metrics (ADE/FDE) if model predicted trajectory
        if trajectory and has_traj:
            import numpy as np
            gt_wps = sample.get("trajectory")
            if gt_wps is not None:
                gt_wps = np.array(gt_wps)
                pred_wps = np.array(trajectory.get("waypoints", []))
                if pred_wps.ndim == 3:
                    pred_wps = pred_wps[0]
                if gt_wps.shape[0] > 0 and pred_wps.shape[0] > 0:
                    min_len = min(len(gt_wps), len(pred_wps))
                    ade = float(np.mean(np.linalg.norm(
                        gt_wps[:min_len, :2] - pred_wps[:min_len, :2], axis=1)))
                    fde = float(np.linalg.norm(
                        gt_wps[min_len - 1, :2] - pred_wps[min_len - 1, :2]))
                    lines.append(f"| **ADE** | {ade:.2f} m |")
                    lines.append(f"| **FDE** | {fde:.2f} m |")
                    # Store in trajectory dict for BEV plot
                    trajectory["gt_waypoints"] = gt_wps.tolist()

        # Trajectory summary
        if has_traj:
            ego_xyz = sample.get("ego_history_xyz")
            if ego_xyz is not None:
                import numpy as np
                ego_xyz = np.array(ego_xyz)
                total_dist = float(np.sum(np.linalg.norm(
                    np.diff(ego_xyz[:, :2], axis=0), axis=1)))
                lines.append(f"| **Track Length** | {total_dist:.1f} m ({len(ego_xyz)} pts) |")

        # Model reasoning (if ran)
        if model_reasoning:
            lines.extend([
                "",
                "---",
                "### Model Reasoning\n",
                model_reasoning,
            ])

        # Pre-annotated CoC reasoning from parquet
        events = sample.get("events", [])
        if events:
            coc_parts = []
            if isinstance(events, (list, tuple)):
                for i, evt in enumerate(events, 1):
                    if isinstance(evt, dict):
                        coc = evt.get("coc", "")
                        frame = evt.get("event_start_frame", "?")
                        ts = evt.get("event_start_timestamp", "?")
                        coc_parts.append(
                            f"#### Event {i} — Frame {frame}\n*Timestamp: {ts}*\n\n{coc}"
                        )
                    elif isinstance(evt, str):
                        coc_parts.append(f"#### Event {i}\n{evt}")

            if coc_parts:
                cluster = sample.get("event_cluster", "")
                lines.extend([
                    "",
                    "---",
                    "### Chain-of-Causation (annotated)\n",
                ])
                if cluster:
                    lines.append(f"**Event Cluster:** {cluster}\n")
                lines.append("\n\n".join(coc_parts))

        return {
            "model": MODEL_INFO[self.model_key]["name"],
            "reasoning_trace": "\n".join(lines),
            "trajectory": trajectory,
            "timestamp": datetime.now().isoformat(),
            "source": "physical_ai_av_sdk",
            "has_camera": has_camera,
        }

    def run_vqa(self, data_sample, question: str) -> dict:
        """Run Visual Question Answering (Alpamayo 1.5 only).

        Args:
            data_sample: A data sample with driving scene images.
            question: Natural language question about the scene.

        Returns:
            Dictionary with the model's answer.
        """
        if self.model_key != "alpamayo-1.5":
            raise ValueError("VQA is only supported by Alpamayo 1.5.")
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        print(f"Running VQA: {question}")
        result = self.model.generate_text(data_sample, question=question)

        return {
            "model": MODEL_INFO[self.model_key]["name"],
            "question": question,
            "answer": result.get("text", ""),
            "timestamp": datetime.now().isoformat(),
        }

    def _format_trajectory(self, trajectory) -> dict | None:
        """Format trajectory output for serialization."""
        if trajectory is None:
            return None

        if isinstance(trajectory, torch.Tensor):
            return {
                "waypoints": trajectory.cpu().numpy().tolist(),
                "horizon_seconds": 6.4,
                "frequency_hz": 10,
                "num_waypoints": trajectory.shape[-2] if trajectory.dim() >= 2 else 0,
            }
        return trajectory

    def save_result(self, result: dict, output_path: str | None = None) -> str:
        """Save inference result to a JSON file.

        Args:
            result: Inference result dictionary.
            output_path: Optional custom output path.

        Returns:
            Path to the saved file.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"result_{ts}.json"
        else:
            output_path = Path(output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"Result saved to: {output_path}")
        return str(output_path)
