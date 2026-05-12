"""Gradio-based GUI for VLAM-Alpamayo."""

import gradio as gr

from src.config import AppConfig, load_config
from src.model_loader import get_model_list, MODEL_IDS, MODEL_INFO
from src.data_loader import get_dataset_info, load_sample_data, DATASETS, DATASET_KEYS
from src.inference import InferenceEngine
from src.visualization import render_result_video, render_trajectory_plot

# ── Force dark mode via JS (overrides system preference) ──────────────────────
FORCE_DARK_JS = """
() => {
    document.querySelector('body').classList.add('dark');
    document.documentElement.style.setProperty('color-scheme', 'dark');
}
"""

# ── NVIDIA green dark theme ───────────────────────────────────────────────────
_nvidia_green = gr.themes.Color(
    c50="#f7fee7", c100="#ecfccb", c200="#d9f99d", c300="#bef264",
    c400="#a3e635", c500="#76b900", c600="#65a30d", c700="#4d7c0f",
    c800="#3f6212", c900="#365314", c950="#1a2e05",
)

DARK_THEME = gr.themes.Default(
    primary_hue=_nvidia_green,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    body_text_color="*neutral_200",
    body_text_color_dark="*neutral_200",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_700",
    block_border_color_dark="*neutral_700",
    block_label_text_color="*neutral_300",
    block_label_text_color_dark="*neutral_300",
    block_radius="12px",
    block_shadow="none",
    input_background_fill="*neutral_800",
    input_background_fill_dark="*neutral_800",
    input_border_color="*neutral_600",
    input_border_color_dark="*neutral_600",
    input_border_width="1px",
    input_radius="8px",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_dark="*primary_500",
    button_primary_text_color="white",
    button_secondary_background_fill="*neutral_700",
    button_secondary_background_fill_dark="*neutral_700",
    button_secondary_border_color="*neutral_600",
    button_secondary_text_color="*neutral_200",
)

# ── Custom CSS — only for custom elements, not base Gradio components ─────────
CUSTOM_CSS = """
.gradio-container { max-width: 100% !important; width: 100% !important; padding: 0 2% !important; }

/* ── Header ──────────────────────────────────────────────── */
.header-container {
    background: linear-gradient(135deg, #0a1628 0%, #122240 40%, #1a3a5c 100%);
    border-radius: 16px;
    padding: 36px 44px;
    margin-bottom: 28px;
    border: 1px solid rgba(118,185,0,0.2);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 60px rgba(118,185,0,0.05);
    position: relative;
    overflow: hidden;
}
.header-container::before {
    content: '';
    position: absolute; top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(118,185,0,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.header-container h1 {
    color: #fff !important; font-size: 2.4em !important;
    font-weight: 800 !important; margin-bottom: 6px !important;
}
.header-container .subtitle {
    color: #8b9dc3 !important; font-size: 1.05em !important;
    margin: 0 0 18px !important; line-height: 1.5;
}
.header-badges { display: flex; gap: 10px; flex-wrap: wrap; }
.header-badge {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px; padding: 5px 16px; color: #c8d6e5;
    font-size: 0.8em; font-weight: 500;
}
.header-badge-green {
    background: rgba(118,185,0,0.15); border-color: rgba(118,185,0,0.35); color: #a3d977;
}
.header-badge-nvidia {
    background: rgba(118,185,0,0.08); border-color: rgba(118,185,0,0.2);
    color: #76b900; font-weight: 700;
}

/* ── Section titles ──────────────────────────────────────── */
.section-title {
    font-size: 0.72em !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 1.5px;
    color: #76b900 !important; margin-bottom: 14px !important;
    padding-bottom: 10px; border-bottom: 1px solid var(--neutral-700);
}

/* ── Run button (NVIDIA green gradient) ──────────────────── */
.run-btn {
    background: linear-gradient(135deg, #76b900 0%, #5a9e00 100%) !important;
    border: none !important; color: #fff !important;
    font-weight: 700 !important; font-size: 1.05em !important;
    border-radius: 10px !important; padding: 14px 0 !important;
    box-shadow: 0 4px 20px rgba(118,185,0,0.25) !important;
    transition: all 0.2s ease !important;
}
.run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(118,185,0,0.35) !important;
    background: linear-gradient(135deg, #84d100 0%, #66b300 100%) !important;
}

/* ── Status left-border accent ───────────────────────────── */
.status-idle textarea { border-left: 3px solid var(--neutral-600) !important; }

/* ── Output description ──────────────────────────────────── */
.output-description { color: var(--neutral-400) !important; font-size: 0.82em !important; margin-top: 8px; }

/* ── VQA info banner ─────────────────────────────────────── */
.vqa-info {
    background: linear-gradient(135deg, #0d1f3c, #122a4a);
    border: 1px solid #1c3a5e; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 18px;
}
.vqa-info strong { color: #58a6ff; }
.vqa-info p { color: #8b9dc3; margin: 6px 0 0; font-size: 0.92em; }

/* ── VQA answer ──────────────────────────────────────────── */
.vqa-answer-box textarea { font-size: 1.02em !important; line-height: 1.7 !important; }

/* ── Table accents ───────────────────────────────────────── */
table th { color: #76b900 !important; text-transform: uppercase; letter-spacing: 0.5px; }
table a { color: #58a6ff !important; }

/* ── Slider accent ───────────────────────────────────────── */
.compact-slider input[type=range] { accent-color: #76b900 !important; }

/* ── Dropdown — dark styling ──────────────────────────────── */
.dark .gr-dropdown, .gradio-container .gr-dropdown {
    background: transparent !important;
    border-color: var(--neutral-600) !important;
}
.dark select, .gradio-container select {
    background: transparent !important;
    color: var(--neutral-200) !important;
    border-color: var(--neutral-600) !important;
}
/* ── Radio button — fully transparent, no background color ── */
.gradio-container .gr-radio,
.gradio-container .gr-radio .wrap,
.gradio-container .gr-radio label,
.gradio-container [role="radiogroup"],
.gradio-container [role="radiogroup"] label,
.gradio-container [role="radiogroup"] > *,
.gradio-container [data-testid="radio"],
.gradio-container [data-testid="radio"] label,
.gradio-container [data-testid="radio"] > div,
.gradio-container .wrap input[type="radio"] + label,
.gradio-container label.svelte-1qxm7ko,
.gradio-container span.svelte-1qxm7ko {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--neutral-200) !important;
}
/* Radio selected state — just text color change, no background */
.gradio-container input[type="radio"]:checked + label,
.gradio-container .selected,
.gradio-container [role="radiogroup"] label.selected,
.gradio-container [role="radiogroup"] label[data-selected],
.gradio-container [role="radiogroup"] label[aria-checked="true"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    color: #76b900 !important;
    font-weight: 700 !important;
}
/* Radio circle accent */
.gradio-container input[type="radio"] {
    accent-color: #76b900 !important;
}
/* Kill any white/light backgrounds on form groups and wrappers */
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .block,
.gradio-container .form {
    background: transparent !important;
    background-color: transparent !important;
}
/* Dropdown option list */
.gradio-container ul[role="listbox"],
.gradio-container .dropdown-options {
    background: var(--neutral-800) !important;
    border-color: var(--neutral-600) !important;
}
.gradio-container ul[role="listbox"] li {
    color: var(--neutral-200) !important;
}
.gradio-container ul[role="listbox"] li:hover,
.gradio-container ul[role="listbox"] li.selected {
    background: rgba(118,185,0,0.15) !important;
    color: #d4e8b0 !important;
}
/* All input labels */
.gradio-container .gr-input-label,
.gradio-container .label-wrap span,
.gradio-container label span {
    color: var(--neutral-300) !important;
}
/* Info text under inputs */
.gradio-container .info-text,
.gradio-container .gr-form > .wrap > span:last-child {
    color: var(--neutral-400) !important;
}

/* ── CoC streaming panel ─────────────────────────────────── */
.coc-panel {
    background: var(--neutral-900) !important;
    border: 1px solid var(--neutral-700) !important;
    border-radius: 12px;
    max-height: 450px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #76b900 var(--neutral-800);
}
.coc-panel::-webkit-scrollbar { width: 6px; }
.coc-panel::-webkit-scrollbar-track { background: var(--neutral-800); }
.coc-panel::-webkit-scrollbar-thumb { background: #76b900; border-radius: 3px; }

/* ── Footer ──────────────────────────────────────────────── */
.footer-container {
    text-align: center; padding: 20px 0 10px; font-size: 0.82em;
    color: var(--neutral-500); border-top: 1px solid var(--neutral-700); margin-top: 28px;
}
.footer-container a { color: #76b900 !important; text-decoration: none; }
.footer-container a:hover { text-decoration: underline; }

/* ── Kill all inline code / highlighted text backgrounds ── */
.gradio-container code,
.gradio-container pre code,
.gradio-container .prose code,
.gradio-container .markdown code,
.gradio-container .coc-panel code,
.gradio-container mark,
.gradio-container .prose mark {
    background: rgba(118,185,0,0.12) !important;
    background-color: rgba(118,185,0,0.12) !important;
    color: #c8e6a0 !important;
    padding: 1px 5px;
    border-radius: 4px;
    border: none !important;
}
/* Remove any highlighted/marked text styling */
.gradio-container .prose *::selection,
.gradio-container .markdown *::selection {
    background: rgba(118,185,0,0.3) !important;
    color: #fff !important;
}
/* Table cells — no highlight backgrounds */
.gradio-container table td,
.gradio-container table td code,
.gradio-container .prose table td code {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--neutral-200) !important;
}
.gradio-container table td code {
    background: rgba(118,185,0,0.1) !important;
    color: #c8e6a0 !important;
}


"""

# ── Global state ──────────────────────────────────────────────────────────────
_engine: InferenceEngine | None = None
_config: AppConfig | None = None
_data_samples: list = []


def _get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def load_model_action(model_key: str) -> str:
    """Load the selected model."""
    global _engine
    try:
        config = _get_config()
        _engine = InferenceEngine(config, model_key=model_key)
        _engine.load()
        info = MODEL_INFO[model_key]
        return (
            f"✅  {info['name']} loaded successfully\n"
            f"     Backbone: {info['backbone']}\n"
            f"     Parameters: {info['params']}\n"
            f"     Features: {', '.join(info['features'])}"
        )
    except Exception as e:
        return f"❌  Error: {e}"


def load_data_action(dataset_key: str, num_samples: int, data_mode: str) -> str:
    """Load sample data from the selected dataset."""
    global _data_samples
    try:
        config = _get_config()
        _data_samples = load_sample_data(
            config,
            dataset_key=dataset_key,
            num_samples=int(num_samples),
            data_mode=data_mode,
        )

        # Build status summary
        n = len(_data_samples)
        lines = [f"✅  {n} sample(s) loaded from '{dataset_key}'"]

        # Show per-sample info
        for i, s in enumerate(_data_samples):
            parts = []
            source = s.get("source", "")

            if source == "physical_ai_av_sdk":
                # SDK clip
                has_video = "camera_front_wide_120fov" in s
                has_ego = "egomotion" in s
                clip_id = s.get("clip_id", "")
                cluster = s.get("event_cluster", "")
                if has_video:
                    nf = len(s["camera_front_wide_120fov"]) if isinstance(s.get("camera_front_wide_120fov"), list) else "?"
                    parts.append(f"📹 {nf} frames")
                if has_ego:
                    parts.append("🗺️ egomotion")
                if cluster:
                    parts.append(f"🏷️ {cluster}")
                label = clip_id[:12] + "…" if len(clip_id) > 12 else (clip_id or f"clip-{i}")

            elif source == "hf_streaming":
                # Normalized streaming sample
                images = s.get("images", [])
                question = s.get("question", "")
                answer = s.get("answer", "")
                if images:
                    parts.append(f"🖼️ {len(images)} image(s)")
                if question:
                    parts.append(f"❓ Q: {question[:50]}{'…' if len(question)>50 else ''}")
                if answer:
                    parts.append("✅ has answer")
                meta_id = s.get("metadata", {}).get("id", "")
                label = str(meta_id)[:12] or f"sample-{i}"

            else:
                # Raw / parquet / other
                parts.append("📄 raw data")
                label = f"sample-{i}"

            detail = " · ".join(parts) if parts else "metadata only"
            lines.append(f"  [{i}] {label}  {detail}")

        return "\n".join(lines)
    except Exception as e:
        return f"❌  Error: {e}"


def run_reasoning_action(
    sample_idx: int, num_traj_samples: int,
) -> tuple[str, str | None, object | None, float | None, float | None, str]:
    """Run reasoning inference — returns (text, video_path, trajectory_image, ade, fde, metrics_md).

    Video is rendered from the data sample's visual content independently
    of whether model inference succeeds.  A model is NOT required for
    displaying dataset images/video.
    """
    global _engine, _data_samples

    if not _data_samples:
        return ("⚠️  No data loaded — use the Setup panel to load dataset samples first.", None, None, None, None, "")

    idx = int(sample_idx)
    if idx < 0 or idx >= len(_data_samples):
        return (f"⚠️  Sample index out of range. Available: 0–{len(_data_samples) - 1}", None, None, None, None, "")

    data_sample = _data_samples[idx]
    config = _get_config()

    # ── Run inference (if engine loaded) ──────────────────────────────
    result = None
    text_out = ""
    if _engine is not None:
        try:
            _engine.config.num_traj_samples = int(num_traj_samples)
            result = _engine.run_reasoning(data_sample)
            _engine.save_result(result)
            text_out = _format_reasoning_result(result)
        except Exception as e:
            text_out = f"⚠️  Inference error: {e}"
    else:
        text_out = _format_data_preview(data_sample)

    # ── Render video from visual data (only if sample has images) ──────
    video_path = None
    if _sample_has_visual_data(data_sample):
        try:
            video_path = render_result_video(
                data_sample, result or {}, output_dir=config.output_dir,
            )
        except Exception as e:
            print(f"Video render error: {e}")

    # ── Render trajectory (from model result or egomotion) ─────────
    traj_img = None
    traj_data = None
    if result and result.get("trajectory"):
        traj_data = result["trajectory"]
    elif data_sample.get("trajectory") is not None:
        traj_data = data_sample["trajectory"]

    if traj_data is not None:
        try:
            traj_img = render_trajectory_plot(traj_data)
        except Exception as e:
            print(f"Trajectory render error: {e}")

    # ── Extract metrics ─────────────────────────────────────────────
    min_ade = None
    min_fde = None
    metrics_md = "*No metrics available — run inference with a loaded model.*"
    if result and result.get("trajectory"):
        traj = result["trajectory"]
        min_ade = traj.get("min_ade")
        min_fde = traj.get("min_fde")
        if min_ade is not None or min_fde is not None:
            lines = ["### Trajectory Metrics\n"]
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            if min_ade is not None:
                lines.append(f"| **minADE** | {min_ade:.4f} m |")
            if min_fde is not None:
                lines.append(f"| **minFDE** | {min_fde:.4f} m |")
            lines.append(f"\n- Horizon: {traj.get('horizon_seconds', 6.4):.1f}s")
            lines.append(f"- Waypoints: {traj.get('num_waypoints', '?')} @ {traj.get('frequency_hz', 10)} Hz")
            metrics_md = "\n".join(lines)

    return (text_out, video_path, traj_img, min_ade, min_fde, metrics_md)


def _sample_has_visual_data(sample: dict) -> bool:
    """Check if a data sample contains any image or video content."""
    # Normalized streaming samples with PIL images
    if sample.get("images"):
        return True
    # SDK samples with camera frames
    visual_keys = {"camera_front_wide_120fov", "images", "image", "video",
                    "frames", "frame", "png"}
    for key in visual_keys:
        if key in sample and sample[key] is not None:
            return True
    # Check for any key containing "camera" or "image" with non-None value
    for key, val in sample.items():
        if val is not None and any(k in key.lower() for k in ("camera", "image")):
            return True
    return False


def _format_data_preview(sample: dict) -> str:
    """Format a data sample preview as Markdown when no model is loaded."""
    source = sample.get("source", "unknown")
    lines = ["### Data Preview\n*No model loaded*\n"]

    if source == "physical_ai_av_sdk":
        clip_id = sample.get("clip_id", "")
        has_cam = "camera_front_wide_120fov" in sample
        n_frames = len(sample["camera_front_wide_120fov"]) if has_cam and isinstance(sample.get("camera_front_wide_120fov"), list) else 0
        has_traj = "trajectory" in sample or "ego_history_xyz" in sample

        lines.append(f"| Property | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| **Clip ID** | `{clip_id}` |")
        lines.append(f"| **Camera** | {n_frames} frames |")
        lines.append(f"| **Egomotion** | {'loaded' if 'egomotion' in sample else 'N/A'} |")
        lines.append(f"| **Trajectory** | {'loaded' if has_traj else 'N/A'} |")

        events = sample.get("events", [])
        cluster = sample.get("event_cluster", "")
        if isinstance(events, (list, tuple)) and events:
            lines.append(f"\n---\n### Chain-of-Causation")
            if cluster:
                lines.append(f"**Event Cluster:** {cluster}\n")
            for i, evt in enumerate(events, 1):
                if isinstance(evt, dict):
                    coc = evt.get("coc", "")
                    frame = evt.get("event_start_frame", "?")
                    ts = evt.get("event_start_timestamp", "?")
                    lines.append(f"#### Event {i} — Frame {frame}")
                    lines.append(f"*Timestamp: {ts}*\n")
                    lines.append(f"{coc}\n")

    elif source == "hf_streaming":
        q = sample.get("question", "")
        a = sample.get("answer", "")
        n_img = len(sample.get("images", []))
        if n_img:
            lines.append(f"**Images:** {n_img} frame(s)\n")
        if q:
            lines.append(f"### Question\n{q}\n")
        if a:
            lines.append(f"### Answer\n{a}\n")

    else:
        for k, v in sample.items():
            val_str = str(v)[:150]
            lines.append(f"- **{k}:** {val_str}")

    lines.append("\n> Load a model to run inference on this data.")
    return "\n".join(lines)


def _format_reasoning_result(result: dict) -> str:
    """Format inference result into a Markdown block."""
    lines = [
        f"### {result['model']}",
        f"*{result['timestamp']}*\n",
        "---",
        "### Chain-of-Causation Reasoning\n",
        str(result.get("reasoning_trace", "*(no reasoning trace generated)*")),
    ]

    if result.get("trajectory"):
        traj = result["trajectory"]
        lines.extend([
            "\n---",
            "### Trajectory Prediction\n",
            f"| Property | Value |",
            f"|---|---|",
            f"| **Waypoints** | {traj.get('num_waypoints', 'N/A')} |",
            f"| **Horizon** | {traj.get('horizon_seconds', 'N/A')}s |",
            f"| **Frequency** | {traj.get('frequency_hz', 'N/A')} Hz |",
        ])
        if traj.get("min_ade") is not None:
            lines.append(f"| **minADE** | {traj['min_ade']:.2f} m |")
        if traj.get("min_fde") is not None:
            lines.append(f"| **minFDE** | {traj['min_fde']:.2f} m |")

    return "\n".join(lines)


def run_vqa_action(sample_idx: int, question: str) -> str:
    """Run VQA on the selected sample (Alpamayo 1.5 only)."""
    global _engine, _data_samples
    if _engine is None:
        return "⚠️  No model loaded. Load Alpamayo 1.5 from the Reasoning tab first."
    if not _data_samples:
        return "⚠️  No data loaded. Load dataset samples from the Reasoning tab first."

    idx = int(sample_idx)
    # Clamp to valid range (slider max may exceed loaded samples)
    if idx >= len(_data_samples):
        idx = len(_data_samples) - 1
    if idx < 0:
        idx = 0

    if not question.strip():
        return "⚠️  Please enter a question."

    try:
        result = _engine.run_vqa(_data_samples[idx], question=question)
        return (
            f"Question\n{result['question']}\n\n"
            f"Answer\n{result['answer']}"
        )
    except ValueError as e:
        return f"⚠️  {e}"
    except Exception as e:
        return f"❌  VQA error: {e}"


def _build_model_info_table() -> str:
    """Build a markdown table comparing models."""
    lines = [
        "| | Alpamayo 1 (R1-10B) | Alpamayo 1.5 (10B) |",
        "|---|---|---|",
        "| **Key** | `alpamayo-1` | `alpamayo-1.5` |",
    ]
    m1, m2 = MODEL_INFO["alpamayo-1"], MODEL_INFO["alpamayo-1.5"]
    lines.append(f"| **Backbone** | {m1['backbone']} | {m2['backbone']} |")
    lines.append(f"| **Parameters** | {m1['params']} | {m2['params']} |")

    all_feats = []
    for f in m1["features"] + m2["features"]:
        if f not in all_feats:
            all_feats.append(f)
    for f in all_feats:
        c1 = "✅" if f in m1["features"] else "—"
        c2 = "✅" if f in m2["features"] else "—"
        lines.append(f"| {f} | {c1} | {c2} |")

    lines.append(f"| **Code** | [GitHub]({m1['code_repo']}) | [GitHub]({m2['code_repo']}) |")
    return "\n".join(lines)


def _build_dataset_table() -> str:
    """Build a markdown table of all datasets."""
    lines = [
        "| Key | Description | Size |",
        "|---|---|---|",
    ]
    for ds in get_dataset_info():
        desc = ds["description"][:90] + ("…" if len(ds["description"]) > 90 else "")
        lines.append(f"| [`{ds['key']}`]({ds['url']}) | {desc} | {ds['size']} |")
    return "\n".join(lines)


def build_gui() -> gr.Blocks:
    """Build the professional dark-themed Gradio interface."""

    with gr.Blocks(
        title="VLAM-Alpamayo — Autonomous Driving Reasoning",
    ) as app:

        # ── Header ───────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-container">
            <h1>🏔️ VLAM-Alpamayo</h1>
            <p class="subtitle">
                Autonomous Driving Reasoning &amp; Trajectory Prediction<br>
                powered by NVIDIA Alpamayo Vision-Language-Action Models
            </p>
            <div class="header-badges">
                <span class="header-badge header-badge-nvidia">NVIDIA</span>
                <span class="header-badge header-badge-green">● Online</span>
                <span class="header-badge">Alpamayo 1 &amp; 1.5</span>
                <span class="header-badge">10.5B Parameters</span>
                <span class="header-badge">14 Datasets</span>
                <span class="header-badge">Chain-of-Causation</span>
                <span class="header-badge">BEV Trajectory</span>
            </div>
        </div>
        """)

        # ── Main Tabs ────────────────────────────────────────────────
        with gr.Tabs(elem_classes="main-tabs"):

            # ━━━ TAB 1: Reasoning Inference ━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("🧠  Reasoning", id="tab-reasoning"):
                with gr.Row(equal_height=False):

                    # ── Left: Model & Dataset ─────────────────────────
                    with gr.Column(scale=1, min_width=300):

                        gr.HTML('<p class="section-title">Model Configuration</p>')
                        model_select = gr.Dropdown(
                            choices=list(MODEL_IDS.keys()),
                            value="alpamayo-1.5",
                            label="Model",
                            info="v1.5 recommended — includes VQA, navigation & RL post-training",
                        )
                        load_model_btn = gr.Button(
                            "Load Model",
                            variant="secondary",
                            size="sm",
                        )
                        model_status = gr.Textbox(
                            label="Model Status",
                            interactive=False,
                            lines=4,
                            value="⏳  No model loaded",
                            elem_classes="status-idle",
                        )

                        gr.HTML('<p class="section-title" style="margin-top:24px">Dataset</p>')
                        dataset_select = gr.Dropdown(
                            choices=DATASET_KEYS,
                            value="physical-ai-av",
                            label="Dataset",
                            info="Streamed on-demand from Hugging Face — no full download",
                        )
                        data_mode = gr.Radio(
                            choices=["video", "parquet"],
                            value="video",
                            label="Data Mode",
                            info="Video: download camera clips via SDK | Parquet: text-only CoC annotations",
                        )
                        num_data_samples = gr.Slider(
                            minimum=1, maximum=10, value=1, step=1,
                            label="Samples to Load",
                            elem_classes="compact-slider",
                        )
                        load_data_btn = gr.Button(
                            "Load Data",
                            variant="secondary",
                            size="sm",
                        )
                        data_status = gr.Textbox(
                            label="Data Status",
                            interactive=False,
                            lines=2,
                            value="⏳  No data loaded",
                            elem_classes="status-idle",
                        )

                    # ── Middle: Inference Controls ────────────────────
                    with gr.Column(scale=1, min_width=280):
                        gr.HTML('<p class="section-title">Inference</p>')
                        sample_idx = gr.Slider(
                            minimum=0, maximum=9, value=0, step=1,
                            label="Sample Index",
                            elem_classes="compact-slider",
                        )
                        num_traj = gr.Slider(
                            minimum=1, maximum=16, value=1, step=1,
                            label="Trajectory Samples",
                            info="More samples → better coverage, higher VRAM usage",
                            elem_classes="compact-slider",
                        )

                        run_btn = gr.Button(
                            "▶  Run Reasoning",
                            variant="primary",
                            size="lg",
                            elem_classes="run-btn",
                        )

                        gr.HTML('<p class="section-title" style="margin-top:24px">Chain-of-Causation</p>')
                        result_output = gr.Markdown(
                            value="*No reasoning output yet — load data and click Run.*",
                            elem_classes="coc-panel",
                        )

                    # ── Right: Visual Output ──────────────────────────
                    with gr.Column(scale=2):
                        gr.HTML('<p class="section-title">Output</p>')

                        # ── Video Section ─────────────────────────────
                        result_video = gr.Video(
                            label="Annotated Driving Video",
                            interactive=False,
                            autoplay=True,
                            height=420,
                        )
                        gr.HTML(
                            '<p class="output-description">Multi-camera composite '
                            'with BEV trajectory overlay, reasoning text & timeline.</p>'
                        )

                        # ── Trajectory + Metrics (side by side) ───────
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.HTML('<p class="section-title">Trajectory (BEV)</p>')
                                result_traj_img = gr.Image(
                                    label="Bird's-Eye-View Trajectory",
                                    interactive=True,
                                    height=350,
                                )
                            with gr.Column(scale=1, min_width=220):
                                gr.HTML('<p class="section-title">Metrics</p>')
                                metrics_ade = gr.Number(
                                    label="minADE (m)",
                                    value=None,
                                    interactive=False,
                                    precision=4,
                                )
                                metrics_fde = gr.Number(
                                    label="minFDE (m)",
                                    value=None,
                                    interactive=False,
                                    precision=4,
                                )
                                metrics_summary = gr.Markdown(
                                    value="*Run inference to see trajectory metrics.*",
                                    elem_classes="coc-panel",
                                )

                # Wire events
                load_model_btn.click(
                    load_model_action,
                    inputs=[model_select],
                    outputs=[model_status],
                )
                load_data_btn.click(
                    load_data_action,
                    inputs=[dataset_select, num_data_samples, data_mode],
                    outputs=[data_status],
                )
                run_btn.click(
                    run_reasoning_action,
                    inputs=[sample_idx, num_traj],
                    outputs=[result_output, result_video, result_traj_img, metrics_ade, metrics_fde, metrics_summary],
                )

            # ━━━ TAB 2: Visual QA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("❓  Visual QA", id="tab-vqa"):
                gr.HTML("""
                <div class="vqa-info">
                    <strong>ℹ️  Visual Question Answering</strong>
                    <p>Ask natural-language questions about a loaded driving scene.
                       Requires <strong>Alpamayo 1.5</strong> — load it from the
                       Reasoning tab first.</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        gr.HTML('<p class="section-title">Input</p>')
                        vqa_sample_idx = gr.Slider(
                            minimum=0, maximum=9, value=0, step=1,
                            label="Sample Index",
                            elem_classes="compact-slider",
                        )
                        vqa_question = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What is the ego vehicle doing?\nIs it safe to change lanes?",
                            lines=3,
                        )
                        vqa_btn = gr.Button(
                            "Ask Question",
                            variant="primary",
                            size="lg",
                            elem_classes="run-btn",
                        )

                    with gr.Column(scale=2):
                        gr.HTML('<p class="section-title">Response</p>')
                        vqa_output = gr.Textbox(
                            label="Model Answer",
                            interactive=False,
                            lines=16,
                            elem_classes="vqa-answer-box",
                        )

                vqa_btn.click(
                    run_vqa_action,
                    inputs=[vqa_sample_idx, vqa_question],
                    outputs=[vqa_output],
                )

            # ━━━ TAB 3: Reference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("📋  Reference", id="tab-info"):
                gr.HTML('<p class="section-title">Model Comparison</p>')
                gr.Markdown(_build_model_info_table())

                gr.HTML('<p class="section-title" style="margin-top:28px">Supported Datasets (14)</p>')
                gr.Markdown(_build_dataset_table())

        # ── Footer ───────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer-container">
            VLAM-Alpamayo &nbsp;·&nbsp; Built with
            <a href="https://www.gradio.app" target="_blank">Gradio</a>
            &nbsp;·&nbsp; Powered by
            <a href="https://huggingface.co/nvidia/Alpamayo-1.5-10B" target="_blank">NVIDIA Alpamayo</a>
            &nbsp;·&nbsp; Author: Chong Kiat Lim
        </div>
        """)

    return app


def launch_gui(config: AppConfig | None = None, share: bool = False):
    """Launch the Gradio GUI application."""
    global _config
    if config:
        _config = config
    else:
        _config = load_config()

    app = build_gui()
    app.launch(
        server_name=_config.gui_host,
        server_port=_config.gui_port,
        share=share,
        theme=DARK_THEME,
        css=CUSTOM_CSS,
        js=FORCE_DARK_JS,
    )


if __name__ == "__main__":
    launch_gui()
