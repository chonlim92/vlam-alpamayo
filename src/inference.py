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

        If the sample already contains pre-annotated reasoning (e.g. from the
        ood_reasoning parquet), return that directly.  Otherwise, run the model.

        Args:
            data_sample: A data sample — either a model-ready dict with camera
                         images, or a parquet row with pre-annotated reasoning.

        Returns:
            Dictionary with reasoning traces and trajectory predictions.
        """
        # ── Check if this is a pre-annotated parquet row ─────────────
        if self._is_parquet_row(data_sample):
            return self._format_parquet_result(data_sample)

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
        # Parquet rows from ood_reasoning.parquet have clip_uuid / event_cluster
        parquet_keys = {"clip_uuid", "event_cluster", "coc", "reasoning"}
        has_parquet_fields = bool(parquet_keys & set(k.lower() for k in sample.keys()))
        # Model-ready samples would have camera/image data
        camera_keys = {"camera_front_wide_120fov", "images", "image", "video", "frames"}
        has_camera = bool(camera_keys & set(k.lower() for k in sample.keys()))
        return has_parquet_fields and not has_camera

    def _format_parquet_result(self, sample: dict) -> dict:
        """Format a pre-annotated parquet row as an inference result."""
        keys = list(sample.keys())
        print(f"[DEBUG] Parquet sample keys: {keys}")
        for k in keys:
            v = sample[k]
            preview = str(v)[:200] if v is not None else "None"
            print(f"  {k}: {preview}")

        # Find the reasoning text — try all plausible column names
        reasoning = ""
        for key in sample:
            kl = key.lower()
            if any(t in kl for t in ("coc", "reasoning", "chain", "annotation", "description", "text", "label")):
                val = sample[key]
                if val is not None and str(val).strip():
                    reasoning = str(val)
                    break

        # Find clip identifier
        clip_id = "unknown"
        for key in sample:
            kl = key.lower()
            if any(t in kl for t in ("clip_uuid", "clip_id", "uuid", "clip", "id", "name")):
                clip_id = str(sample[key])
                break

        lines = [f"Clip ID:         {clip_id}"]

        # Include all other metadata fields
        skip_keys = set()
        for key in sample:
            kl = key.lower()
            if any(t in kl for t in ("clip_uuid", "clip_id", "uuid", "clip", "id", "name",
                                      "coc", "reasoning", "chain", "annotation", "description", "text", "label")):
                skip_keys.add(key)

        for key in sample:
            if key not in skip_keys:
                val = sample[key]
                if val is not None:
                    val_str = str(val)
                    if len(val_str) <= 200:
                        lines.append(f"{key}:  {val_str}")

        lines.extend([
            "",
            "━━━  Chain-of-Causation (pre-annotated)  ━━━━━━━━━━━━━━━━",
            "",
            reasoning if reasoning else "(no reasoning text found in this row)",
        ])

        return {
            "model": f"{MODEL_INFO[self.model_key]['name']} (pre-annotated data)",
            "reasoning_trace": "\n".join(lines),
            "trajectory": None,
            "timestamp": datetime.now().isoformat(),
            "source": "parquet_annotation",
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
