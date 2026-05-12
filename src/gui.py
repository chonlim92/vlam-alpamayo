"""Gradio-based GUI for VLAM-Alpamayo."""

import gradio as gr

from src.config import AppConfig, load_config
from src.model_loader import get_model_list, MODEL_IDS, MODEL_INFO
from src.data_loader import get_dataset_info, load_sample_data, DATASETS, DATASET_KEYS
from src.inference import InferenceEngine
from src.visualization import render_result_video, render_trajectory_plot

# ── Custom CSS — Dark Professional Theme ──────────────────────────────────────
CUSTOM_CSS = """
/* ── Global overrides ────────────────────────────────────── */
body, .gradio-container {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
}
.gradio-container {
    max-width: 1400px !important;
}

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
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(118,185,0,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.header-container h1 {
    color: #ffffff !important;
    font-size: 2.4em !important;
    font-weight: 800 !important;
    margin-bottom: 6px !important;
    letter-spacing: -0.5px;
}
.header-container .subtitle {
    color: #8b9dc3 !important;
    font-size: 1.05em !important;
    margin: 0 0 18px !important;
    line-height: 1.5;
}
.header-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.header-badge {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 5px 16px;
    color: #c8d6e5;
    font-size: 0.8em;
    font-weight: 500;
    backdrop-filter: blur(4px);
}
.header-badge-green {
    background: rgba(118,185,0,0.15);
    border-color: rgba(118,185,0,0.35);
    color: #a3d977;
}
.header-badge-nvidia {
    background: rgba(118,185,0,0.08);
    border-color: rgba(118,185,0,0.2);
    color: #76b900;
    font-weight: 700;
}

/* ── Panels & blocks ─────────────────────────────────────── */
.block, .gr-block, .gr-box, .gr-panel,
div[class*="block"], div[class*="panel"] {
    background: #161b22 !important;
    border-color: #21262d !important;
    border-radius: 12px !important;
}

/* ── Section headings ────────────────────────────────────── */
.section-title {
    font-size: 0.72em !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #76b900 !important;
    margin-bottom: 14px !important;
    padding-bottom: 10px;
    border-bottom: 1px solid #21262d;
}

/* ── Inputs ──────────────────────────────────────────────── */
input, textarea, select, .gr-input, .gr-dropdown {
    background: #0d1117 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #76b900 !important;
    box-shadow: 0 0 0 2px rgba(118,185,0,0.15) !important;
}
label, .gr-label {
    color: #8b949e !important;
    font-weight: 600 !important;
    font-size: 0.88em !important;
}

/* ── Status textboxes ────────────────────────────────────── */
.status-idle textarea {
    border-left: 3px solid #30363d !important;
    background: #0d1117 !important;
    color: #8b949e !important;
}
.status-loaded textarea {
    border-left: 3px solid #76b900 !important;
    background: #0d1117 !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.run-btn {
    background: linear-gradient(135deg, #76b900 0%, #5a9e00 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    letter-spacing: 0.5px;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(118,185,0,0.25) !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
.run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(118,185,0,0.35) !important;
    background: linear-gradient(135deg, #84d100 0%, #66b300 100%) !important;
}
.secondary-btn {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
.secondary-btn:hover {
    background: #30363d !important;
    border-color: #484f58 !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
.main-tabs button, .output-tabs button {
    background: transparent !important;
    color: #8b949e !important;
    font-weight: 600 !important;
    font-size: 0.92em !important;
    padding: 10px 22px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s ease !important;
}
.main-tabs button.selected, .output-tabs button.selected {
    color: #76b900 !important;
    border-bottom: 2px solid #76b900 !important;
    background: rgba(118,185,0,0.05) !important;
}
.main-tabs button:hover, .output-tabs button:hover {
    color: #c9d1d9 !important;
    background: rgba(255,255,255,0.03) !important;
}

/* ── Output areas ────────────────────────────────────────── */
.output-tabs {
    border: 1px solid #21262d;
    border-radius: 12px;
    overflow: hidden;
    background: #0d1117 !important;
}
.output-description {
    color: #6e7681 !important;
    font-size: 0.82em !important;
    margin-top: 8px;
}

/* ── VQA info banner ─────────────────────────────────────── */
.vqa-info {
    background: linear-gradient(135deg, #0d1f3c, #122a4a);
    border: 1px solid #1c3a5e;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 18px;
}
.vqa-info strong {
    color: #58a6ff;
}
.vqa-info p {
    color: #8b9dc3;
    margin: 6px 0 0;
    font-size: 0.92em;
}

/* ── VQA answer ──────────────────────────────────────────── */
.vqa-answer-box textarea {
    font-size: 1.02em !important;
    line-height: 1.7 !important;
    background: #0d1117 !important;
    color: #e6edf3 !important;
}

/* ── Tables in Reference tab ─────────────────────────────── */
table {
    width: 100% !important;
    border-collapse: collapse !important;
}
table th {
    background: #161b22 !important;
    color: #76b900 !important;
    font-weight: 700 !important;
    font-size: 0.85em !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 12px 16px !important;
    border-bottom: 2px solid #21262d !important;
}
table td {
    padding: 10px 16px !important;
    border-bottom: 1px solid #21262d !important;
    color: #c9d1d9 !important;
    font-size: 0.9em;
}
table tr:hover td {
    background: rgba(118,185,0,0.03) !important;
}
table a {
    color: #58a6ff !important;
}

/* ── Sliders ─────────────────────────────────────────────── */
.compact-slider input[type=range] {
    height: 4px !important;
    accent-color: #76b900 !important;
}
.compact-slider .progress {
    background: #76b900 !important;
}

/* ── Footer ──────────────────────────────────────────────── */
.footer-container {
    text-align: center;
    padding: 20px 0 10px;
    color: #484f58;
    font-size: 0.82em;
    border-top: 1px solid #21262d;
    margin-top: 28px;
}
.footer-container a {
    color: #76b900 !important;
    text-decoration: none;
}
.footer-container a:hover {
    text-decoration: underline;
}

/* ── Markdown in dark mode ───────────────────────────────── */
.prose, .markdown-text, .gr-markdown {
    color: #c9d1d9 !important;
}
.prose h1, .prose h2, .prose h3 {
    color: #e6edf3 !important;
}
.prose code {
    background: #21262d !important;
    color: #76b900 !important;
    padding: 2px 6px;
    border-radius: 4px;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0d1117;
}
::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #484f58;
}

/* ── Video player ────────────────────────────────────────── */
video {
    border-radius: 8px !important;
    border: 1px solid #21262d !important;
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


def load_data_action(dataset_key: str, num_samples: int) -> str:
    """Load sample data from the selected dataset."""
    global _data_samples
    try:
        config = _get_config()
        _data_samples = load_sample_data(
            config, dataset_key=dataset_key, num_samples=int(num_samples),
        )
        ds_name = DATASETS[dataset_key]["description"][:80]
        return f"✅  {len(_data_samples)} sample(s) loaded from '{dataset_key}'\n     {ds_name}…"
    except Exception as e:
        return f"❌  Error: {e}"


def run_reasoning_action(
    sample_idx: int, num_traj_samples: int,
) -> tuple[str, str | None, object | None]:
    """Run reasoning inference — returns (text, video_path, trajectory_image)."""
    global _engine, _data_samples

    if _engine is None:
        return ("⚠️  No model loaded — use the Setup panel to load a model first.", None, None)
    if not _data_samples:
        return ("⚠️  No data loaded — use the Setup panel to load dataset samples first.", None, None)

    idx = int(sample_idx)
    if idx < 0 or idx >= len(_data_samples):
        return (f"⚠️  Sample index out of range. Available: 0–{len(_data_samples) - 1}", None, None)

    try:
        _engine.config.num_traj_samples = int(num_traj_samples)
        result = _engine.run_reasoning(_data_samples[idx])
        _engine.save_result(result)

        text_out = _format_reasoning_result(result)

        config = _get_config()
        video_path = render_result_video(
            _data_samples[idx], result, output_dir=config.output_dir,
        )
        traj_img = render_trajectory_plot(result.get("trajectory"))

        return (text_out, video_path, traj_img)
    except Exception as e:
        return (f"❌  Inference error: {e}", None, None)


def _format_reasoning_result(result: dict) -> str:
    """Format inference result into a readable text block."""
    lines = [
        f"Model          {result['model']}",
        f"Timestamp      {result['timestamp']}",
        "",
        "━━━  Chain-of-Causation Reasoning  ━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        result.get("reasoning_trace", "(no reasoning trace generated)"),
    ]

    if result.get("trajectory"):
        traj = result["trajectory"]
        lines.extend([
            "",
            "━━━  Trajectory Prediction  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"Waypoints      {traj.get('num_waypoints', 'N/A')}",
            f"Horizon        {traj.get('horizon_seconds', 'N/A')}s @ {traj.get('frequency_hz', 'N/A')} Hz",
        ])

    return "\n".join(lines)


def run_vqa_action(sample_idx: int, question: str) -> str:
    """Run VQA on the selected sample (Alpamayo 1.5 only)."""
    global _engine, _data_samples
    if _engine is None:
        return "⚠️  No model loaded. Load Alpamayo 1.5 from the Reasoning tab first."
    if not _data_samples:
        return "⚠️  No data loaded. Load dataset samples from the Reasoning tab first."

    idx = int(sample_idx)
    if idx < 0 or idx >= len(_data_samples):
        return "⚠️  Sample index out of range."

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

                    # ── Left: Setup Panel ─────────────────────────────
                    with gr.Column(scale=1, min_width=340):

                        gr.HTML('<p class="section-title">Model Configuration</p>')
                        model_select = gr.Dropdown(
                            choices=list(MODEL_IDS.keys()),
                            value="alpamayo-1.5",
                            label="Model",
                            info="v1.5 recommended — includes VQA, navigation & RL post-training",
                        )
                        load_model_btn = gr.Button(
                            "Load Model",
                            variant="primary",
                            size="sm",
                            elem_classes="secondary-btn",
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
                        num_data_samples = gr.Slider(
                            minimum=1, maximum=10, value=1, step=1,
                            label="Samples to Load",
                            elem_classes="compact-slider",
                        )
                        load_data_btn = gr.Button(
                            "Load Data",
                            size="sm",
                            elem_classes="secondary-btn",
                        )
                        data_status = gr.Textbox(
                            label="Data Status",
                            interactive=False,
                            lines=2,
                            value="⏳  No data loaded",
                            elem_classes="status-idle",
                        )

                        gr.HTML('<p class="section-title" style="margin-top:24px">Inference</p>')
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

                    # ── Right: Results Panel ──────────────────────────
                    with gr.Column(scale=2):
                        gr.HTML('<p class="section-title">Output</p>')

                        with gr.Tabs(elem_classes="output-tabs"):
                            with gr.Tab("📹  Video", id="out-video"):
                                result_video = gr.Video(
                                    label="Annotated Driving Video",
                                    interactive=False,
                                    autoplay=True,
                                    height=500,
                                )
                                gr.HTML(
                                    '<p class="output-description">Multi-camera composite '
                                    'with BEV trajectory overlay, reasoning text & timeline.</p>'
                                )

                            with gr.Tab("🗺️  Trajectory", id="out-traj"):
                                result_traj_img = gr.Image(
                                    label="Bird's-Eye-View Trajectory",
                                    interactive=False,
                                    height=500,
                                )
                                gr.HTML(
                                    '<p class="output-description">Predicted 6.4s future trajectory '
                                    '(64 waypoints @ 10 Hz) in ego-vehicle BEV coordinates.</p>'
                                )

                            with gr.Tab("📝  Reasoning Trace", id="out-text"):
                                result_output = gr.Textbox(
                                    label="Chain-of-Causation Output",
                                    interactive=False,
                                    lines=26,
                                )

                # Wire events
                load_model_btn.click(
                    load_model_action,
                    inputs=[model_select],
                    outputs=[model_status],
                )
                load_data_btn.click(
                    load_data_action,
                    inputs=[dataset_select, num_data_samples],
                    outputs=[data_status],
                )
                run_btn.click(
                    run_reasoning_action,
                    inputs=[sample_idx, num_traj],
                    outputs=[result_output, result_video, result_traj_img],
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
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    launch_gui()
