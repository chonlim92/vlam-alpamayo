"""Gradio-based GUI for VLAM-Alpamayo."""

import gradio as gr

from src.config import AppConfig, load_config
from src.model_loader import get_model_list, MODEL_IDS, MODEL_INFO
from src.data_loader import get_dataset_info, load_sample_data, DATASETS, DATASET_KEYS
from src.inference import InferenceEngine
from src.visualization import render_result_video, render_trajectory_plot

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* Header */
.header-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.header-container h1 {
    color: #ffffff !important;
    font-size: 2.2em !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
    letter-spacing: -0.5px;
}
.header-container p {
    color: #94a3b8 !important;
    font-size: 1.05em !important;
    margin: 0 !important;
}
.header-badges {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.header-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 4px 14px;
    color: #e2e8f0;
    font-size: 0.82em;
    font-weight: 500;
}
.header-badge-green {
    background: rgba(34,197,94,0.15);
    border-color: rgba(34,197,94,0.3);
    color: #86efac;
}

/* Section panels */
.panel-section {
    background: #fafbfc;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
}
.dark .panel-section {
    background: #1e293b;
    border-color: #334155;
}

/* Section headings */
.section-title {
    font-size: 0.78em !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}
.dark .section-title {
    color: #94a3b8 !important;
    border-bottom-color: #334155;
}

/* Status boxes */
.status-ready {
    border-left: 4px solid #22c55e !important;
}
.status-idle {
    border-left: 4px solid #94a3b8 !important;
}

/* Run button */
.run-btn {
    background: linear-gradient(135deg, #76b900 0%, #5a9e00 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    letter-spacing: 0.5px;
    border-radius: 10px !important;
    padding: 12px 0 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(118,185,0,0.3) !important;
}
.run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(118,185,0,0.4) !important;
}

/* Output tabs */
.output-tabs {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
}
.dark .output-tabs {
    border-color: #334155;
}

/* Info cards */
.info-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: box-shadow 0.2s ease;
}
.info-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.dark .info-card {
    background: #1e293b;
    border-color: #334155;
}
.dark .info-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

/* Footer */
.footer-container {
    text-align: center;
    padding: 16px 0 8px;
    color: #94a3b8;
    font-size: 0.82em;
    border-top: 1px solid #e5e7eb;
    margin-top: 24px;
}
.dark .footer-container {
    border-top-color: #334155;
}

/* VQA chat-style */
.vqa-answer-box textarea {
    font-size: 1.02em !important;
    line-height: 1.7 !important;
}

/* Tabs styling */
.main-tabs > .tab-nav > button {
    font-weight: 600 !important;
    font-size: 0.95em !important;
    padding: 10px 20px !important;
}

/* Compact sliders */
.compact-slider input[type=range] {
    height: 6px !important;
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


def load_model_action(model_key: str) -> tuple[str, str]:
    """Load the selected model. Returns (status_text, status_class)."""
    global _engine
    try:
        config = _get_config()
        _engine = InferenceEngine(config, model_key=model_key)
        _engine.load()
        info = MODEL_INFO[model_key]
        status = (
            f"✅  {info['name']} loaded successfully\n"
            f"     Backbone: {info['backbone']}\n"
            f"     Parameters: {info['params']}\n"
            f"     Features: {', '.join(info['features'])}"
        )
        return status
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
        return ("⚠️  No model loaded — use the Setup panel on the left to load a model first.", None, None)
    if not _data_samples:
        return ("⚠️  No data loaded — use the Setup panel on the left to load dataset samples first.", None, None)

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
        return "⚠️  No model loaded. Please load Alpamayo 1.5 from the Reasoning tab first."
    if not _data_samples:
        return "⚠️  No data loaded. Please load dataset samples from the Reasoning tab first."

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


# ── Theme ─────────────────────────────────────────────────────────────────────
alpamayo_theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f0fdf4", c100="#dcfce7", c200="#bbf7d0", c300="#86efac",
        c400="#4ade80", c500="#76b900", c600="#5a9e00", c700="#4d7c0f",
        c800="#3f6212", c900="#365314", c950="#1a2e05",
    ),
    secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_border_color="*neutral_200",
        block_radius="12px",
        block_shadow="0 1px 3px rgba(0,0,0,0.04)",
        input_border_width="1px",
        input_border_color="*neutral_300",
        input_radius="8px",
        button_primary_background_fill="*primary_500",
        button_primary_text_color="white",
        button_secondary_background_fill="*neutral_100",
        button_secondary_border_color="*neutral_300",
    )


def build_gui() -> gr.Blocks:
    """Build the professional Gradio interface."""

    with gr.Blocks(
        title="VLAM-Alpamayo — Autonomous Driving Reasoning",
    ) as app:

        # ── Header ───────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-container">
            <h1>🏔️ VLAM-Alpamayo</h1>
            <p>Autonomous Driving Reasoning &amp; Trajectory Prediction
               powered by NVIDIA Alpamayo Vision-Language-Action Models</p>
            <div class="header-badges">
                <span class="header-badge header-badge-green">● Ready</span>
                <span class="header-badge">Alpamayo 1 &amp; 1.5</span>
                <span class="header-badge">10.5B Parameters</span>
                <span class="header-badge">14 Datasets</span>
                <span class="header-badge">Chain-of-Causation</span>
            </div>
        </div>
        """)

        # ── Main Tabs ────────────────────────────────────────────────
        with gr.Tabs(elem_classes="main-tabs"):

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 1: Reasoning Inference
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Reasoning", id="tab-reasoning"):
                with gr.Row(equal_height=False):

                    # ── Left: Setup Panel ─────────────────────────────
                    with gr.Column(scale=1, min_width=320):

                        # Model setup
                        gr.HTML('<p class="section-title">Model</p>')
                        model_select = gr.Dropdown(
                            choices=list(MODEL_IDS.keys()),
                            value="alpamayo-1.5",
                            label="Select Model",
                            info="Alpamayo 1.5 recommended — includes VQA & navigation",
                        )
                        load_model_btn = gr.Button(
                            "Load Model",
                            variant="primary",
                            size="sm",
                        )
                        model_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=4,
                            value="No model loaded",
                            elem_classes="status-idle",
                        )

                        # Dataset setup
                        gr.HTML('<p class="section-title" style="margin-top:20px">Dataset</p>')
                        dataset_select = gr.Dropdown(
                            choices=DATASET_KEYS,
                            value="physical-ai-av",
                            label="Select Dataset",
                            info="Streamed on-demand from Hugging Face",
                        )
                        num_data_samples = gr.Slider(
                            minimum=1, maximum=10, value=1, step=1,
                            label="Samples to Load",
                            elem_classes="compact-slider",
                        )
                        load_data_btn = gr.Button("Load Data", size="sm")
                        data_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=2,
                            value="No data loaded",
                            elem_classes="status-idle",
                        )

                        # Inference params
                        gr.HTML('<p class="section-title" style="margin-top:20px">Inference Parameters</p>')
                        sample_idx = gr.Slider(
                            minimum=0, maximum=9, value=0, step=1,
                            label="Sample Index",
                            elem_classes="compact-slider",
                        )
                        num_traj = gr.Slider(
                            minimum=1, maximum=16, value=1, step=1,
                            label="Trajectory Samples",
                            info="More samples = better coverage, higher VRAM",
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
                        gr.HTML('<p class="section-title">Results</p>')

                        with gr.Tabs(elem_classes="output-tabs"):
                            with gr.Tab("Video Output", id="out-video"):
                                result_video = gr.Video(
                                    label="Annotated Driving Video",
                                    interactive=False,
                                    autoplay=True,
                                    height=480,
                                )
                                gr.Markdown(
                                    "<small>Multi-camera composited view with BEV trajectory overlay, "
                                    "reasoning text, and timeline bar.</small>"
                                )

                            with gr.Tab("Trajectory", id="out-traj"):
                                result_traj_img = gr.Image(
                                    label="Bird's-Eye-View Trajectory Plot",
                                    interactive=False,
                                    height=480,
                                )
                                gr.Markdown(
                                    "<small>Predicted future trajectory (6.4s, 64 waypoints @ 10 Hz) "
                                    "in ego-vehicle BEV coordinates.</small>"
                                )

                            with gr.Tab("Reasoning Trace", id="out-text"):
                                result_output = gr.Textbox(
                                    label="Chain-of-Causation Output",
                                    interactive=False,
                                    lines=24,
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

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 2: Visual Question Answering
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Visual QA", id="tab-vqa"):
                gr.HTML("""
                <div style="background:linear-gradient(135deg,#eff6ff,#f0f9ff);
                     border:1px solid #bfdbfe; border-radius:10px;
                     padding:16px 20px; margin-bottom:16px;">
                    <strong style="color:#1e40af">ℹ️  Visual Question Answering</strong>
                    <p style="color:#1e3a5f; margin:6px 0 0; font-size:0.92em">
                        Ask natural-language questions about a loaded driving scene.
                        Requires <strong>Alpamayo 1.5</strong> — load it from the
                        Reasoning tab first.
                    </p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
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
                        gr.HTML('<p class="section-title">Answer</p>')
                        vqa_output = gr.Textbox(
                            label="Model Response",
                            interactive=False,
                            lines=14,
                            elem_classes="vqa-answer-box",
                        )

                vqa_btn.click(
                    run_vqa_action,
                    inputs=[vqa_sample_idx, vqa_question],
                    outputs=[vqa_output],
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 3: Models & Datasets Reference
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Reference", id="tab-info"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<p class="section-title">Model Comparison</p>')
                        gr.Markdown(_build_model_info_table())

                gr.HTML('<p class="section-title" style="margin-top:24px">Supported Datasets (14)</p>')
                gr.Markdown(_build_dataset_table())

        # ── Footer ───────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer-container">
            VLAM-Alpamayo &nbsp;·&nbsp; Built with
            <a href="https://www.gradio.app" target="_blank" style="color:#76b900">Gradio</a>
            &nbsp;·&nbsp; Powered by
            <a href="https://huggingface.co/nvidia/Alpamayo-1.5-10B" target="_blank" style="color:#76b900">NVIDIA Alpamayo</a>
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
        theme=alpamayo_theme,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    launch_gui()
