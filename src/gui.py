"""Gradio-based GUI for VLAM-Alpamayo."""

import gradio as gr

from src.config import AppConfig, load_config
from src.model_loader import get_model_list, MODEL_IDS, MODEL_INFO
from src.data_loader import get_dataset_info, load_sample_data
from src.inference import InferenceEngine

# Global state
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
            f"✅ Loaded: {info['name']}\n"
            f"Backbone: {info['backbone']}\n"
            f"Parameters: {info['params']}\n"
            f"Features: {', '.join(info['features'])}"
        )
    except Exception as e:
        return f"❌ Error loading model: {e}"


def load_data_action(num_samples: int) -> str:
    """Load sample data from the dataset."""
    global _data_samples
    try:
        config = _get_config()
        _data_samples = load_sample_data(config, num_samples=int(num_samples))
        return f"✅ Loaded {len(_data_samples)} sample(s) from PhysicalAI-AV dataset."
    except Exception as e:
        return f"❌ Error loading data: {e}"


def run_reasoning_action(sample_idx: int, num_traj_samples: int) -> str:
    """Run reasoning inference on the selected sample."""
    global _engine, _data_samples
    if _engine is None:
        return "❌ Please load a model first."
    if not _data_samples:
        return "❌ Please load data samples first."

    idx = int(sample_idx)
    if idx < 0 or idx >= len(_data_samples):
        return f"❌ Invalid sample index. Available: 0 to {len(_data_samples) - 1}"

    try:
        _engine.config.num_traj_samples = int(num_traj_samples)
        result = _engine.run_reasoning(_data_samples[idx])
        _engine.save_result(result)

        output_lines = [
            f"🧠 Model: {result['model']}",
            f"📅 Timestamp: {result['timestamp']}",
            "",
            "═══ Chain-of-Causation Reasoning ═══",
            result.get("reasoning_trace", "(no trace)"),
        ]

        if result.get("trajectory"):
            traj = result["trajectory"]
            output_lines.extend([
                "",
                "═══ Trajectory Prediction ═══",
                f"Waypoints: {traj.get('num_waypoints', 'N/A')}",
                f"Horizon: {traj.get('horizon_seconds', 'N/A')}s at {traj.get('frequency_hz', 'N/A')}Hz",
            ])

        return "\n".join(output_lines)
    except Exception as e:
        return f"❌ Inference error: {e}"


def run_vqa_action(sample_idx: int, question: str) -> str:
    """Run VQA on the selected sample (Alpamayo 1.5 only)."""
    global _engine, _data_samples
    if _engine is None:
        return "❌ Please load a model first."
    if not _data_samples:
        return "❌ Please load data samples first."

    idx = int(sample_idx)
    if idx < 0 or idx >= len(_data_samples):
        return f"❌ Invalid sample index."

    if not question.strip():
        return "❌ Please enter a question."

    try:
        result = _engine.run_vqa(_data_samples[idx], question=question)
        return (
            f"❓ Question: {result['question']}\n\n"
            f"💡 Answer: {result['answer']}"
        )
    except ValueError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ VQA error: {e}"


def get_model_info_md() -> str:
    """Build markdown for model info display."""
    lines = ["## Available Models\n"]
    for m in get_model_list():
        lines.append(f"### {m['name']}")
        lines.append(f"- **Key:** `{m['key']}`")
        lines.append(f"- **Backbone:** {m['backbone']}")
        lines.append(f"- **Parameters:** {m['params']}")
        lines.append(f"- **Features:** {', '.join(m['features'])}")
        lines.append(f"- **Code:** [{m['code_repo']}]({m['code_repo']})")
        lines.append("")
    return "\n".join(lines)


def get_dataset_info_md() -> str:
    """Build markdown for dataset info display."""
    lines = ["## Recommended Datasets\n"]
    for ds in get_dataset_info():
        lines.append(f"### {ds['key']}")
        lines.append(f"- {ds['description']}")
        lines.append(f"- **Size:** {ds['size']}")
        lines.append(f"- **URL:** [{ds['url']}]({ds['url']})")
        lines.append("")
    return "\n".join(lines)


def build_gui() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="VLAM-Alpamayo",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# 🏔️ VLAM-Alpamayo\n"
            "### Autonomous Driving Reasoning with NVIDIA Alpamayo Models\n"
            "Chain-of-Causation reasoning and trajectory prediction for autonomous driving scenes."
        )

        with gr.Tabs():
            # --- Tab 1: Inference ---
            with gr.Tab("🧠 Reasoning Inference"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_select = gr.Dropdown(
                            choices=list(MODEL_IDS.keys()),
                            value="alpamayo-1.5",
                            label="Select Model",
                        )
                        load_model_btn = gr.Button("Load Model", variant="primary")
                        model_status = gr.Textbox(label="Model Status", interactive=False, lines=5)

                        gr.Markdown("---")

                        num_data_samples = gr.Slider(
                            minimum=1, maximum=10, value=1, step=1,
                            label="Number of Data Samples",
                        )
                        load_data_btn = gr.Button("Load Data")
                        data_status = gr.Textbox(label="Data Status", interactive=False)

                    with gr.Column(scale=2):
                        sample_idx = gr.Slider(
                            minimum=0, maximum=9, value=0, step=1,
                            label="Sample Index",
                        )
                        num_traj = gr.Slider(
                            minimum=1, maximum=16, value=1, step=1,
                            label="Trajectory Samples",
                        )
                        run_btn = gr.Button("▶ Run Reasoning", variant="primary", size="lg")
                        result_output = gr.Textbox(
                            label="Reasoning Output",
                            interactive=False,
                            lines=20,
                        )

                load_model_btn.click(load_model_action, inputs=[model_select], outputs=[model_status])
                load_data_btn.click(load_data_action, inputs=[num_data_samples], outputs=[data_status])
                run_btn.click(
                    run_reasoning_action,
                    inputs=[sample_idx, num_traj],
                    outputs=[result_output],
                )

            # --- Tab 2: VQA ---
            with gr.Tab("❓ Visual QA (v1.5 only)"):
                gr.Markdown(
                    "Ask natural language questions about a driving scene. "
                    "**Requires Alpamayo 1.5 to be loaded.**"
                )
                with gr.Row():
                    vqa_sample_idx = gr.Slider(
                        minimum=0, maximum=9, value=0, step=1,
                        label="Sample Index",
                    )
                    vqa_question = gr.Textbox(
                        label="Question",
                        placeholder="e.g., What is the ego vehicle doing? Is it safe to change lanes?",
                    )
                vqa_btn = gr.Button("Ask", variant="primary")
                vqa_output = gr.Textbox(label="Answer", interactive=False, lines=10)

                vqa_btn.click(
                    run_vqa_action,
                    inputs=[vqa_sample_idx, vqa_question],
                    outputs=[vqa_output],
                )

            # --- Tab 3: Info ---
            with gr.Tab("ℹ️ Model & Dataset Info"):
                gr.Markdown(get_model_info_md())
                gr.Markdown("---")
                gr.Markdown(get_dataset_info_md())

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
    )


if __name__ == "__main__":
    launch_gui()
