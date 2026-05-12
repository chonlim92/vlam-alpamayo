"""Inference engine for Alpamayo models."""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
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

        # Note: non-SDK samples need custom preprocessing; this is a placeholder
        # for future support of other data formats.
        try:
            pred_xyz, pred_rot, extra = (
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=data_sample,
                    num_traj_samples=self.config.num_traj_samples,
                    return_extra=True,
                )
            )
            reasoning = extra.get("cot", [""])[0] if extra else ""
            # Ensure reasoning is a string (model may return ndarray)
            if not isinstance(reasoning, str):
                reasoning = str(reasoning.tolist() if hasattr(reasoning, 'tolist') else reasoning)
            trajectory = self._format_trajectory(pred_xyz)
        except (ValueError, TypeError) as e:
            reasoning = f"(inference failed — data format may be incompatible: {e})"
            trajectory = None

        output = {
            "model": MODEL_INFO[self.model_key]["name"],
            "reasoning_trace": reasoning,
            "trajectory": trajectory,
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
        gt_future_xyz = None
        if self.model is not None and has_camera:
            try:
                model_reasoning, trajectory, gt_future_xyz = self._run_sdk_inference(
                    clip_id, sample,
                )
            except Exception as e:
                model_reasoning = f"(model inference failed: {e})"
                print(f"  Model inference error: {e}")
                import traceback
                traceback.print_exc()

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
        if trajectory and trajectory.get("min_ade") is not None:
            lines.append(f"| **minADE** | {trajectory['min_ade']:.2f} m |")
        if trajectory and trajectory.get("min_fde") is not None:
            lines.append(f"| **minFDE** | {trajectory['min_fde']:.2f} m |")

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
                str(model_reasoning) if not isinstance(model_reasoning, str) else model_reasoning,
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
            "model_reasoning": model_reasoning or "",
            "trajectory": trajectory,
            "timestamp": datetime.now().isoformat(),
            "source": "physical_ai_av_sdk",
            "has_camera": has_camera,
        }

    def run_vqa(self, data_sample, question: str) -> dict:
        """Run Visual Question Answering (Alpamayo 1.5 only).

        Uses the SDK helper to build a chat message with the question,
        then generates text via the model's standard inference pipeline.

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

        from alpamayo1_5 import helper

        # Build chat messages from frames, then append question
        clip_id = data_sample.get("clip_id")
        if clip_id and data_sample.get("source") == "physical_ai_av_sdk":
            # SDK sample — use load_physical_aiavdataset for model-ready data
            from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
            data = load_physical_aiavdataset(clip_id)
            messages = helper.create_message(
                frames=data["image_frames"].flatten(0, 1),
                camera_indices=data["camera_indices"],
            )
            messages.append({"role": "user", "content": question})
        else:
            # Fallback: try to build messages from images in the sample
            images = data_sample.get("images", [])
            if not images and "camera_front_wide_120fov" in data_sample:
                images = data_sample["camera_front_wide_120fov"]

            if not images:
                # No images — build a text-only VQA
                messages = [{"role": "user", "content": question}]
            else:
                # Convert PIL images / numpy arrays to torch tensors for the SDK helper
                import torch
                from torchvision.transforms.functional import to_tensor
                from PIL import Image as _PILImage

                tensor_frames = []
                for img in images:
                    if isinstance(img, np.ndarray):
                        # HWC uint8 numpy → CHW float tensor
                        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    elif isinstance(img, _PILImage.Image):
                        t = to_tensor(img)
                    elif isinstance(img, torch.Tensor):
                        t = img
                    else:
                        continue
                    tensor_frames.append(t)

                if tensor_frames:
                    frames_tensor = torch.stack(tensor_frames)
                    messages = helper.create_message(
                        frames=frames_tensor,
                    )
                    # Append the user question to the conversation
                    messages.append({"role": "user", "content": question})
                else:
                    messages = [{"role": "user", "content": question}]

        # Debug: log message structure to diagnose any issues
        print(f"  VQA messages: {len(messages)} entries")
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                content = msg.get("content", "")
                content_desc = f"str({len(content)})" if isinstance(content, str) else f"{type(content).__name__}"
                if isinstance(content, list):
                    content_desc = f"list({len(content)}): [{', '.join(type(c).__name__ for c in content[:3])}...]"
                print(f"    [{i}] role={msg.get('role')}, content={content_desc}")
            else:
                print(f"    [{i}] type={type(msg).__name__}, value={str(msg)[:80]}")

        # Sanitize: ensure all messages are dicts with role/content
        clean_messages = []
        for msg in messages:
            if isinstance(msg, str):
                # Wrap bare strings as user messages
                clean_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                clean_messages.append(msg)
            else:
                print(f"  WARNING: skipping unexpected message type: {type(msg)}")
        messages = clean_messages

        # Tokenize
        processor = helper.get_processor(self.model.tokenizer)
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError as e:
            # Likely the processor can't handle multimodal content blocks
            # Fall back to text-only: extract text from content blocks
            print(f"  apply_chat_template failed: {e}")
            print(f"  Falling back to text-only chat template...")
            text_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text parts from multimodal content blocks
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = " ".join(text_parts) if text_parts else ""
                text_messages.append({"role": msg.get("role", "user"), "content": content})
            inputs = processor.apply_chat_template(
                text_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        # Generate text
        import torch
        inputs = helper.to_device(inputs, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Decode — skip input tokens
        input_len = inputs["input_ids"].shape[1]
        answer = self.model.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True,
        ).strip()

        return {
            "model": MODEL_INFO[self.model_key]["name"],
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        }

    def _run_sdk_inference(
        self, clip_id: str, sample: dict,
    ) -> tuple[str, dict | None, np.ndarray | None]:
        """Run model inference on an SDK sample using the proper Alpamayo pipeline.

        Uses the SDK's ``load_physical_aiavdataset`` to prepare model-ready input
        (tokenized chat, ego history tensors), then calls the model with the
        correct API: ``sample_trajectories_from_data_with_vlm_rollout``.

        Returns:
            (reasoning_text, trajectory_dict, gt_future_xyz)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        print(f"  Preparing model input for clip {clip_id}...")

        # ── Load model-ready data via SDK ─────────────────────────────
        if self.model_key == "alpamayo-1.5":
            from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
            from alpamayo1_5 import helper
        else:
            from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
            from alpamayo_r1 import helper

        data = load_physical_aiavdataset(clip_id)
        print(f"  SDK data loaded: {list(data.keys())}")

        # ── Build chat messages from camera frames ────────────────────
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )

        # ── Tokenize ──────────────────────────────────────────────────
        processor = helper.get_processor(self.model.tokenizer)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        # ── Build model inputs ────────────────────────────────────────
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")

        # ── Run inference ─────────────────────────────────────────────
        print(f"  Running model inference (num_traj_samples={self.config.num_traj_samples})...")
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = (
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=self.config.num_traj_samples,
                    max_generation_length=256,
                    return_extra=True,
                )
            )

        # ── Extract Chain-of-Causation reasoning ─────────────────────
        cot_list = extra.get("cot", []) if extra else []
        reasoning = cot_list[0] if cot_list else "(no reasoning generated)"
        # Ensure reasoning is a string (may be numpy array or tensor)
        if not isinstance(reasoning, str):
            try:
                if hasattr(reasoning, 'item'):
                    reasoning = str(reasoning.item())
                elif hasattr(reasoning, 'tolist'):
                    val = reasoning.tolist()
                    reasoning = val if isinstance(val, str) else str(val)
                else:
                    reasoning = str(reasoning)
            except (ValueError, TypeError):
                reasoning = str(reasoning)
        print(f"  CoC reasoning: {len(reasoning)} chars")

        # ── Format trajectory ─────────────────────────────────────────
        # pred_xyz shape: (batch, n_group, n_traj_samples, n_future_steps, 3)
        pred_np = pred_xyz.cpu().numpy()
        # Squeeze batch/group dims → (n_traj_samples, n_future_steps, 3)
        while pred_np.ndim > 3:
            pred_np = pred_np[0]
        # For BEV plot: primary trajectory is [0] → (n_future_steps, 2)
        primary_xy = pred_np[0, :, :2]

        trajectory = {
            "waypoints": primary_xy.tolist(),
            "all_samples": pred_np[:, :, :2].tolist() if pred_np.shape[0] > 1 else None,
            "horizon_seconds": 6.4,
            "frequency_hz": 10,
            "num_waypoints": primary_xy.shape[0],
        }

        # ── Compute minADE / minFDE against ground truth ─────────────
        gt_future_xyz = None
        if "ego_future_xyz" in data:
            gt_xy = data["ego_future_xyz"].cpu().numpy()
            while gt_xy.ndim > 2:
                gt_xy = gt_xy[0]
            gt_future_xyz = gt_xy
            gt_xy_2d = gt_xy[:, :2]  # (n_future_steps, 2)

            # pred_np: (n_traj_samples, n_future_steps, 3)
            pred_xy_all = pred_np[:, :, :2]  # (n_traj_samples, n_future_steps, 2)
            n_steps = min(gt_xy_2d.shape[0], pred_xy_all.shape[1])
            # ADE per sample: mean displacement over timesteps
            displacements = np.linalg.norm(
                pred_xy_all[:, :n_steps] - gt_xy_2d[None, :n_steps], axis=2,
            )  # (n_traj_samples, n_steps)
            ade_per_sample = displacements.mean(axis=1)
            fde_per_sample = np.linalg.norm(
                pred_xy_all[:, n_steps - 1] - gt_xy_2d[n_steps - 1], axis=1,
            )
            trajectory["min_ade"] = float(ade_per_sample.min())
            trajectory["min_fde"] = float(fde_per_sample.min())
            trajectory["gt_waypoints"] = gt_xy_2d.tolist()
            print(f"  minADE: {trajectory['min_ade']:.2f}m, minFDE: {trajectory['min_fde']:.2f}m")

        return reasoning, trajectory, gt_future_xyz

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
