"""Visualization utilities for driving scenes and trajectory predictions."""

import math
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


# ── Colour palette (RGB) ─────────────────────────────────────────────────────
_TRAJ_COLOR = (0, 220, 100)       # green – predicted trajectory
_TRAJ_ALT_COLOR = (50, 200, 255)  # cyan  – alternative trajectory samples
_EGO_COLOR = (0, 100, 255)        # blue  – ego marker
_TEXT_BG = (30, 30, 30)
_TEXT_FG = (255, 255, 255)


def render_result_video(
    data_sample: dict,
    result: dict,
    output_dir: str = "output",
    fps: int = 5,
) -> str:
    """Render an annotated MP4 video from a data sample and inference result.

    Composites the front-camera frames with an overlay showing:
    - The Chain-of-Causation reasoning text (scrolling, large font)
    - The predicted trajectory drawn on a BEV mini-map
    - Trajectory drawn on the camera view (projected from BEV)
    - A timeline bar

    Frames are expected in RGB order. Output video is H.264 (browser-compatible).

    Args:
        data_sample: Raw data sample (must contain camera images).
        result: Inference result dict from InferenceEngine.
        output_dir: Directory to write the video file.
        fps: Frames per second of the output video (default 5 for readability).

    Returns:
        Absolute path to the generated .mp4 file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = output_path / f"result_{ts}.mp4"

    frames = _extract_frames(data_sample)
    if not frames:
        # No camera data – generate a static summary frame as a short clip
        frames = [_make_placeholder_frame()]

    # Upscale small frames for readability
    min_w = 1280
    if frames[0].shape[1] < min_w:
        scale = min_w / frames[0].shape[1]
        frames = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4) for f in frames]

    h, w = frames[0].shape[:2]
    reasoning_text = result.get("reasoning_trace", "")
    model_name = result.get("model", "Alpamayo")

    # Get trajectory from result or data sample
    trajectory_waypoints = None
    gt_waypoints = None
    traj_metrics = {}
    if result.get("trajectory") and "waypoints" in result["trajectory"]:
        trajectory_waypoints = np.array(result["trajectory"]["waypoints"])
        if result["trajectory"].get("gt_waypoints"):
            gt_waypoints = np.array(result["trajectory"]["gt_waypoints"])
        if result["trajectory"].get("min_ade") is not None:
            traj_metrics["min_ade"] = result["trajectory"]["min_ade"]
        if result["trajectory"].get("min_fde") is not None:
            traj_metrics["min_fde"] = result["trajectory"]["min_fde"]
    elif data_sample.get("trajectory") is not None:
        trajectory_waypoints = np.array(data_sample["trajectory"])

    # Build annotated frames
    annotated = []
    total = len(frames)
    reasoning_lines = _wrap_text(reasoning_text, max_chars=80)
    # Scroll reasoning text across frames
    lines_per_frame = max(1, len(reasoning_lines) // max(total, 1))

    for idx, frame in enumerate(frames):
        # Work in RGB
        canvas = frame.copy()

        # ── BEV trajectory mini-map (top-right) ──────────────────────────
        if trajectory_waypoints is not None:
            wps = trajectory_waypoints
            if wps.ndim == 3:
                wps = wps[0]
            _draw_bev_minimap(canvas, wps, idx, total, gt_waypoints=gt_waypoints)
            # Draw predicted trajectory on camera view (green)
            _draw_trajectory_on_camera(canvas, wps, idx, total)
            # Draw GT trajectory on camera view (orange) if available
            if gt_waypoints is not None:
                _draw_trajectory_on_camera(
                    canvas, gt_waypoints, idx, total,
                    color_start=(255, 140, 0), color_fade=0.5,
                )
            # Per-frame trajectory info (top-left)
            _draw_frame_trajectory_info(canvas, wps, idx, total, metrics=traj_metrics)

        # ── Reasoning text overlay (bottom) ──────────────────────────────
        visible_start = min(idx * lines_per_frame, max(len(reasoning_lines) - 6, 0))
        visible_lines = reasoning_lines[visible_start:visible_start + 6]
        _draw_text_overlay(canvas, visible_lines, model_name)

        # ── Timeline bar (very bottom) ───────────────────────────────────
        _draw_timeline(canvas, idx, total)

        annotated.append(canvas)

    # ── Write video (frames are RGB) ────────────────────────────────────
    video_path = _write_browser_compatible_video(annotated, video_path, fps, w, h)

    return str(video_path)


def render_trajectory_plot(trajectory: dict | np.ndarray | None) -> np.ndarray | None:
    """Render a standalone BEV trajectory plot as an image (numpy array).

    Args:
        trajectory: Trajectory dict with 'waypoints' key, or raw (N,2) array.

    Returns:
        RGB numpy array (plot image) or None.
    """
    if trajectory is None:
        return None

    # Accept raw numpy array
    if isinstance(trajectory, np.ndarray):
        waypoints = trajectory
    elif isinstance(trajectory, dict) and "waypoints" in trajectory:
        waypoints = np.array(trajectory["waypoints"])
    else:
        return None

    size = 600
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)

    if waypoints.ndim == 3:
        # Multiple trajectory samples – draw all
        for s in range(waypoints.shape[0]):
            _draw_traj_on_bev(img, waypoints[s], size, color=_TRAJ_ALT_COLOR, thickness=1)
        _draw_traj_on_bev(img, waypoints[0], size, color=_TRAJ_COLOR, thickness=2)
    elif waypoints.ndim == 2:
        _draw_traj_on_bev(img, waypoints, size, color=_TRAJ_COLOR, thickness=2)
    else:
        return None

    # Draw ego vehicle marker at center
    cx, cy = size // 2, int(size * 0.75)
    cv2.circle(img, (cx, cy), 10, _EGO_COLOR, -1)
    cv2.putText(img, "EGO", (cx - 18, cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_FG, 1)

    # Axis labels
    cv2.putText(img, "BEV Trajectory", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _TEXT_FG, 2)

    if isinstance(trajectory, dict):
        horizon = trajectory.get("horizon_seconds", "?")
        freq = trajectory.get("frequency_hz", "?")
        n_wp = trajectory.get("num_waypoints", waypoints.shape[0] if waypoints.ndim == 2 else "?")
    else:
        n_wp = waypoints.shape[0]
        horizon = "?"
        freq = "?"

    cv2.putText(
        img, f"{n_wp} pts | {horizon}s @ {freq}Hz",
        (10, size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )

    # ADE / FDE metrics (if ground truth available)
    if isinstance(trajectory, dict) and trajectory.get("gt_waypoints") is not None:
        gt = np.array(trajectory["gt_waypoints"])
        # Draw GT trajectory in orange
        _draw_traj_on_bev(img, gt, size, color=(255, 140, 0), thickness=2)
        pred = waypoints[0] if waypoints.ndim == 3 else waypoints
        if gt.shape[0] > 0 and pred.shape[0] > 0:
            min_len = min(len(gt), len(pred))
            ade = np.mean(np.linalg.norm(gt[:min_len, :2] - pred[:min_len, :2], axis=1))
            fde = np.linalg.norm(gt[min_len - 1, :2] - pred[min_len - 1, :2])
            cv2.putText(
                img, f"ADE: {ade:.2f}m  FDE: {fde:.2f}m",
                (10, size - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 50), 1,
            )
        # Legend
        cv2.putText(img, "Pred", (size - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TRAJ_COLOR, 1)
        cv2.putText(img, "GT", (size - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)

    return img


# ── Internal helpers ──────────────────────────────────────────────────────────


def _extract_frames(data_sample: dict) -> list[np.ndarray]:
    """Extract camera frames from a data sample.

    Supports common data formats: list of PIL images, numpy arrays,
    torch tensors, or video bytes.
    """
    frames = []

    # Try front-wide camera first, then any available camera
    for cam_key in ("camera_front_wide_120fov", "front_wide", "camera_front_wide", "images", "video"):
        raw = data_sample.get(cam_key)
        if raw is not None:
            frames = _convert_to_frames(raw)
            if frames:
                return frames

    # Try top-level image/images field
    for key in ("image", "frame", "frames", "png"):
        raw = data_sample.get(key)
        if raw is not None:
            frames = _convert_to_frames(raw)
            if frames:
                return frames

    # Fallback: look for any key containing "camera" or "image"
    for key, val in data_sample.items():
        if any(k in key.lower() for k in ("camera", "image", "frame")):
            frames = _convert_to_frames(val)
            if frames:
                return frames

    return frames


def _convert_to_frames(raw) -> list[np.ndarray]:
    """Convert various data types to a list of RGB numpy frames."""
    frames = []

    # PIL Image — already RGB
    try:
        from PIL import Image
        if isinstance(raw, Image.Image):
            arr = np.array(raw.convert('RGB'))
            frames.append(arr)
            return frames
    except ImportError:
        pass

    # numpy array – single frame or batch (assume RGB)
    if isinstance(raw, np.ndarray):
        if raw.ndim == 3:
            if raw.shape[2] == 3:
                frames.append(raw.copy())
            return frames
        elif raw.ndim == 4:
            for i in range(raw.shape[0]):
                frames.append(raw[i].copy())
            return frames

    # torch tensor
    try:
        import torch
        if isinstance(raw, torch.Tensor):
            arr = raw.cpu().numpy()
            if arr.ndim == 4:
                # (B, C, H, W) or (B, H, W, C)
                if arr.shape[1] in (1, 3):
                    arr = arr.transpose(0, 2, 3, 1)
                for i in range(arr.shape[0]):
                    f = (arr[i] * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr[i].astype(np.uint8)
                    frames.append(f)
            elif arr.ndim == 3:
                if arr.shape[0] in (1, 3):
                    arr = arr.transpose(1, 2, 0)
                f = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                frames.append(f)
            return frames
    except ImportError:
        pass

    # list of items – recurse
    if isinstance(raw, (list, tuple)):
        for item in raw:
            frames.extend(_convert_to_frames(item))
        return frames

    return frames


def _make_placeholder_frame(w: int = 960, h: int = 540) -> np.ndarray:
    """Generate a dark placeholder frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)
    cv2.putText(
        frame, "No camera data available",
        (w // 2 - 180, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2,
    )
    return frame


def _draw_bev_minimap(
    canvas: np.ndarray,
    waypoints: np.ndarray,
    frame_idx: int,
    total_frames: int,
    map_size: int = 220,
    gt_waypoints: np.ndarray | None = None,
) -> None:
    """Draw a BEV trajectory mini-map in the top-right corner."""
    h, w = canvas.shape[:2]
    margin = 15
    x0, y0 = w - map_size - margin, margin

    # Semi-transparent background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + map_size, y0 + map_size), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + map_size, y0 + map_size), (80, 80, 80), 1)

    wps = waypoints
    if wps.ndim != 2 or wps.shape[0] == 0:
        return

    # Draw GT first (underneath) in orange
    if gt_waypoints is not None and gt_waypoints.ndim == 2 and gt_waypoints.shape[0] > 0:
        _draw_traj_on_bev_region(
            canvas, gt_waypoints, x0, y0, map_size, frame_idx, total_frames,
            color_base=(255, 140, 0),
        )

    # Draw predicted on top in green
    _draw_traj_on_bev_region(canvas, wps, x0, y0, map_size, frame_idx, total_frames)

    # Legend
    cv2.putText(canvas, "Pred", (x0 + 5, y0 + map_size - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, _TRAJ_COLOR, 1, cv2.LINE_AA)
    if gt_waypoints is not None:
        cv2.putText(canvas, "GT", (x0 + 5, y0 + map_size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 140, 0), 1, cv2.LINE_AA)


def _draw_traj_on_bev_region(
    canvas: np.ndarray,
    wps: np.ndarray,
    x0: int, y0: int, size: int,
    frame_idx: int, total_frames: int,
    color_base: tuple | None = None,
) -> None:
    """Draw trajectory points in a BEV region on the canvas."""
    # Use x, y columns (first two)
    xs = wps[:, 0]
    ys = wps[:, 1] if wps.shape[1] > 1 else np.zeros_like(xs)

    # Normalize to fit mini-map with padding
    pad = 20
    all_coords = np.stack([xs, ys], axis=1)
    mn = all_coords.min(axis=0)
    mx = all_coords.max(axis=0)
    span = mx - mn
    span[span == 0] = 1.0

    # Map to pixel coords in the minimap
    px = ((xs - mn[0]) / span[0] * (size - 2 * pad) + pad + x0).astype(int)
    py = ((ys - mn[1]) / span[1] * (size - 2 * pad) + pad + y0).astype(int)
    # Flip y so forward is up
    py = y0 + size - (py - y0)

    # Draw trajectory line
    pts = np.stack([px, py], axis=1)
    for i in range(len(pts) - 1):
        progress = i / max(len(pts) - 1, 1)
        if color_base is not None:
            alpha = max(0.4, 1.0 - progress * 0.6)
            color = (
                int(color_base[0] * alpha),
                int(color_base[1] * alpha),
                int(color_base[2] * alpha),
            )
        else:
            color = (
                int(0 + 255 * progress),
                int(255 - 100 * progress),
                int(128 - 128 * progress),
            )
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), color, 2)

    # Ego marker
    ego_x = x0 + size // 2
    ego_y = y0 + size - pad
    cv2.circle(canvas, (ego_x, ego_y), 5, _EGO_COLOR, -1)

    # Animate: show current position on trajectory based on frame progress
    if total_frames > 1:
        traj_idx = min(int(frame_idx / total_frames * len(pts)), len(pts) - 1)
        marker_color = color_base if color_base else (0, 0, 255)
        cv2.circle(canvas, tuple(pts[traj_idx]), 4, marker_color, -1)


def _draw_traj_on_bev(
    img: np.ndarray,
    wps: np.ndarray,
    size: int,
    color: tuple = _TRAJ_COLOR,
    thickness: int = 2,
) -> None:
    """Draw trajectory on a standalone BEV image."""
    if wps.ndim != 2 or wps.shape[0] == 0:
        return

    xs = wps[:, 0]
    ys = wps[:, 1] if wps.shape[1] > 1 else np.zeros_like(xs)

    pad = 40
    mn_x, mx_x = xs.min(), xs.max()
    mn_y, mx_y = ys.min(), ys.max()
    span_x = mx_x - mn_x if mx_x != mn_x else 1.0
    span_y = mx_y - mn_y if mx_y != mn_y else 1.0

    cx, cy_base = size // 2, int(size * 0.75)
    scale = (size - 2 * pad) / max(span_x, span_y)

    px = (cx + (xs - xs.mean()) * scale).astype(int)
    py = (cy_base - (ys - ys[0]) * scale).astype(int)

    pts = np.stack([px, py], axis=1)
    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)


def _draw_trajectory_on_camera(
    canvas: np.ndarray,
    wps: np.ndarray,
    frame_idx: int,
    total_frames: int,
    color_start: tuple = (0, 220, 100),
    color_fade: float = 0.7,
) -> None:
    """Draw ego trajectory projected onto the camera view.

    Simple BEV→camera projection: maps ego x,y to a perspective-like
    projection on the lower half of the frame. Forward (y+) goes up.
    """
    h, w = canvas.shape[:2]
    if wps.ndim != 2 or wps.shape[0] < 2:
        return

    xs = wps[:, 0]
    ys = wps[:, 1] if wps.shape[1] > 1 else np.zeros_like(xs)

    # Normalize: center x, normalize y to 0..1 range
    x_range = max(abs(xs.max() - xs.min()), 1.0)
    y_range = max(ys.max() - ys.min(), 1.0)

    # Project: near bottom of frame (y=0) to vanishing point area (y_max)
    cx = w // 2
    vanish_y = int(h * 0.35)  # vanishing point height
    bottom_y = int(h * 0.92)  # bottom of road

    pts = []
    for i in range(len(xs)):
        # Depth factor: 0 (near) to 1 (far)
        depth = (ys[i] - ys[0]) / y_range if y_range > 0 else 0
        depth = max(0, min(1, depth))

        # Y on screen: near=bottom_y, far=vanish_y
        py = int(bottom_y - depth * (bottom_y - vanish_y))
        # X on screen: perspective narrowing with depth
        perspective = 1.0 - 0.7 * depth
        lateral = (xs[i] - xs.mean()) / x_range * 2
        px = int(cx + lateral * (w * 0.3) * perspective)

        pts.append((px, py))

    # Draw trajectory line
    for i in range(len(pts) - 1):
        progress = i / max(len(pts) - 1, 1)
        alpha = max(0.3, 1.0 - progress * color_fade)
        color = (
            int(color_start[0] * alpha),
            int(color_start[1] * alpha),
            int(color_start[2] * alpha),
        )
        thickness = max(1, int(4 * (1.0 - 0.5 * progress)))
        cv2.line(canvas, pts[i], pts[i + 1], color, thickness)

    # Current position marker (only for primary predicted trajectory)
    if color_start == (0, 220, 100) and total_frames > 1:
        traj_idx = min(int(frame_idx / total_frames * len(pts)), len(pts) - 1)
        cv2.circle(canvas, pts[traj_idx], 6, (255, 80, 0), -1)
        cv2.circle(canvas, pts[traj_idx], 8, (255, 255, 255), 2)


def _draw_text_overlay(canvas: np.ndarray, lines: list[str], model_name: str) -> None:
    """Draw a semi-transparent text overlay at the bottom of the frame."""
    h, w = canvas.shape[:2]
    font_scale = max(0.55, min(0.8, w / 1600))
    line_h = int(28 * font_scale / 0.55)
    panel_h = (len(lines) + 2) * line_h + 15
    y_start = h - panel_h

    # Semi-transparent panel
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, y_start), (w, h), _TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Model name header
    cv2.putText(
        canvas, f"[{model_name}] Chain-of-Causation Reasoning",
        (16, y_start + line_h),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (100, 200, 255), 1, cv2.LINE_AA,
    )

    # Reasoning text — larger, anti-aliased
    for i, line in enumerate(lines):
        y = y_start + (i + 2) * line_h + 5
        cv2.putText(
            canvas, line, (16, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, _TEXT_FG, 1, cv2.LINE_AA,
        )


def _draw_timeline(canvas: np.ndarray, idx: int, total: int) -> None:
    """Draw a thin progress bar at the very bottom."""
    h, w = canvas.shape[:2]
    bar_h = 6
    progress = (idx + 1) / max(total, 1)
    cv2.rectangle(canvas, (0, h - bar_h), (w, h), (60, 60, 60), -1)
    cv2.rectangle(canvas, (0, h - bar_h), (int(w * progress), h), (118, 185, 0), -1)


def _draw_frame_trajectory_info(
    canvas: np.ndarray,
    wps: np.ndarray,
    frame_idx: int,
    total_frames: int,
    metrics: dict | None = None,
) -> None:
    """Draw per-frame trajectory position, speed, and heading on the video."""
    h, w = canvas.shape[:2]
    if wps.ndim != 2 or wps.shape[0] < 2:
        return

    # Map frame index to trajectory index
    traj_idx = min(int(frame_idx / max(total_frames, 1) * len(wps)), len(wps) - 1)
    x, y = wps[traj_idx, 0], wps[traj_idx, 1]

    # Compute instantaneous speed (distance between consecutive points)
    if traj_idx > 0:
        dx = wps[traj_idx, 0] - wps[traj_idx - 1, 0]
        dy = wps[traj_idx, 1] - wps[traj_idx - 1, 1]
        dist = np.sqrt(dx**2 + dy**2)
        # Estimate dt from total trajectory / total points
        total_dist = np.sum(np.linalg.norm(np.diff(wps[:, :2], axis=0), axis=1))
        heading_deg = np.degrees(np.arctan2(dy, dx))
    else:
        dist = 0.0
        total_dist = np.sum(np.linalg.norm(np.diff(wps[:, :2], axis=0), axis=1))
        heading_deg = 0.0

    # Determine box height based on whether metrics are present
    has_metrics = metrics and (metrics.get("min_ade") is not None or metrics.get("min_fde") is not None)
    box_w, box_h = 280, 112 if has_metrics else 90
    margin = 15
    overlay = canvas.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + box_w, margin + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    cv2.rectangle(canvas, (margin, margin), (margin + box_w, margin + box_h), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    c = (200, 200, 200)
    g = (118, 185, 0)
    y_pos = margin + 20

    cv2.putText(canvas, f"Frame {frame_idx+1}/{total_frames}", (margin + 8, y_pos), font, fs, g, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Traj pt {traj_idx+1}/{len(wps)}", (margin + 150, y_pos), font, fs, c, 1, cv2.LINE_AA)
    y_pos += 22
    cv2.putText(canvas, f"Pos: ({x:.1f}, {y:.1f}) m", (margin + 8, y_pos), font, fs, c, 1, cv2.LINE_AA)
    y_pos += 22
    cv2.putText(canvas, f"Heading: {heading_deg:.1f} deg", (margin + 8, y_pos), font, fs, c, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Track: {total_dist:.1f}m total", (margin + 150, y_pos), font, fs, c, 1, cv2.LINE_AA)

    # ADE/FDE metrics line
    if has_metrics:
        y_pos += 22
        ade_str = f"ADE: {metrics['min_ade']:.3f}m" if metrics.get("min_ade") is not None else ""
        fde_str = f"FDE: {metrics['min_fde']:.3f}m" if metrics.get("min_fde") is not None else ""
        cv2.putText(canvas, ade_str, (margin + 8, y_pos), font, fs, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, fde_str, (margin + 150, y_pos), font, fs, (0, 200, 255), 1, cv2.LINE_AA)


def _wrap_text(text: str, max_chars: int = 70) -> list[str]:
    """Word-wrap text into lines of at most max_chars."""
    if not text:
        return ["(no reasoning trace)"]
    words = text.replace("\n", " ").split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines or ["(no reasoning trace)"]


def _reencode_h264(mp4v_path: Path) -> Path:
    """Try to re-encode with H.264 via ffmpeg for browser playback.

    Falls back to the original mp4v file if ffmpeg is not available.
    """
    h264_path = mp4v_path.with_suffix(".h264.mp4")
    try:
        import subprocess
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp4v_path),
                "-c:v", "libx264", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                str(h264_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if h264_path.exists() and h264_path.stat().st_size > 0:
            mp4v_path.unlink()
            return h264_path
        print(f"ffmpeg re-encode produced empty file: {result.stderr[:200]}")
    except FileNotFoundError:
        print(
            "WARNING: ffmpeg not found — video saved with mp4v codec which "
            "browsers cannot play. Install ffmpeg:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  conda: conda install -c conda-forge ffmpeg"
        )
    except subprocess.TimeoutExpired:
        print("WARNING: ffmpeg re-encode timed out")
    except Exception as e:
        print(f"WARNING: ffmpeg re-encode failed: {e}")
    return mp4v_path


def _write_browser_compatible_video(
    frames: list[np.ndarray],
    video_path: Path,
    fps: int,
    w: int,
    h: int,
) -> Path:
    """Write RGB frames to a browser-playable MP4 (H.264).

    Strategy (in priority order):
    1. imageio-ffmpeg — writes H.264 directly via bundled ffmpeg
    2. PyAV (av package) — writes H.264 via libavcodec bindings
    3. OpenCV H.264 — only works if system ffmpeg backend is compiled in
    4. OpenCV mp4v + ffmpeg re-encode — write mp4v then re-encode
    5. OpenCV mp4v fallback — browsers can't play but file is valid

    Input frames are in RGB order.
    """

    # Strategy 1: imageio-ffmpeg (accepts RGB directly)
    try:
        import imageio.v3 as iio
        h264_path = video_path.with_suffix(".mp4")
        with iio.imopen(str(h264_path), "w", plugin="pyav") as writer:
            writer.init_video_stream("libx264", fps=fps)
            for frame in frames:
                writer.write_frame(frame)
        if h264_path.exists() and h264_path.stat().st_size > 0:
            return h264_path
    except Exception:
        pass

    # Strategy 2: PyAV (accepts RGB directly)
    try:
        import av as _av
        h264_path = video_path.with_suffix(".mp4")
        container = _av.open(str(h264_path), mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "fast"}
        for frame in frames:
            av_frame = _av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        if h264_path.exists() and h264_path.stat().st_size > 0:
            return h264_path
    except Exception:
        pass

    # Strategy 3: OpenCV H.264 (needs BGR)
    h264_path = video_path.with_suffix(".mp4")
    for codec in ("avc1", "x264", "H264"):
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(h264_path), fourcc, fps, (w, h))
            if writer.isOpened():
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                if h264_path.exists() and h264_path.stat().st_size > 0:
                    return h264_path
            writer.release()
        except Exception:
            continue

    # Strategy 4: OpenCV mp4v + ffmpeg re-encode (needs BGR)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    return _reencode_h264(video_path)
