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

        Args:
            data_sample: A sample from the PhysicalAI-AV dataset containing
                         multi-camera images, egomotion history, and timestamps.

        Returns:
            Dictionary with reasoning traces and trajectory predictions.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        print("Running Chain-of-Causation reasoning...")

        if self.model_key == "alpamayo-1":
            result = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data_sample,
                num_traj_samples=self.config.num_traj_samples,
            )
        else:
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
