"""CLI application for VLAM-Alpamayo."""

import argparse
import sys
import json

from src.config import load_config
from src.model_loader import get_model_list, MODEL_IDS
from src.data_loader import get_dataset_info, load_sample_data
from src.inference import InferenceEngine


def cmd_info(args):
    """Show information about available models and datasets."""
    print("=" * 60)
    print("VLAM-Alpamayo — NVIDIA Alpamayo Models for Autonomous Driving")
    print("=" * 60)

    print("\n📦 Available Models:\n")
    for model in get_model_list():
        print(f"  [{model['key']}] {model['name']}")
        print(f"    Backbone:   {model['backbone']}")
        print(f"    Parameters: {model['params']}")
        print(f"    Features:   {', '.join(model['features'])}")
        print(f"    Code:       {model['code_repo']}")
        print()

    print("📊 Recommended Datasets:\n")
    for ds in get_dataset_info():
        print(f"  [{ds['key']}]")
        print(f"    {ds['description']}")
        print(f"    Size: {ds['size']}")
        print(f"    URL:  {ds['url']}")
        print()


def cmd_run(args):
    """Run inference on driving scene data."""
    config = load_config()

    # Override from CLI args
    model_key = args.model or config.default_model
    if model_key not in MODEL_IDS:
        print(f"Error: Unknown model '{model_key}'. Choose from: {list(MODEL_IDS.keys())}")
        sys.exit(1)

    if args.samples:
        config.num_traj_samples = args.samples

    print(f"Initializing inference with model: {model_key}")
    engine = InferenceEngine(config, model_key=model_key)
    engine.load()

    # Load sample data
    num_data = args.num_data or 1
    print(f"Loading {num_data} sample(s) from PhysicalAI-AV dataset...")
    data_samples = load_sample_data(config, num_samples=num_data)

    if not data_samples:
        print("Error: No data samples loaded. Check your HF token and dataset access.")
        sys.exit(1)

    results = []
    for i, sample in enumerate(data_samples):
        print(f"\n--- Sample {i + 1}/{len(data_samples)} ---")
        result = engine.run_reasoning(sample)
        results.append(result)

        # Print reasoning trace
        print(f"\nReasoning Trace:")
        print(result.get("reasoning_trace", "(no trace)"))

        if result.get("trajectory"):
            traj = result["trajectory"]
            print(f"\nTrajectory: {traj['num_waypoints']} waypoints over {traj['horizon_seconds']}s")

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = None

    if len(results) == 1:
        engine.save_result(results[0], output_path)
    else:
        engine.save_result({"results": results}, output_path)


def cmd_vqa(args):
    """Run Visual Question Answering on a driving scene."""
    config = load_config()

    engine = InferenceEngine(config, model_key="alpamayo-1.5")
    engine.load()

    print("Loading sample data...")
    data_samples = load_sample_data(config, num_samples=1)

    if not data_samples:
        print("Error: No data samples loaded.")
        sys.exit(1)

    result = engine.run_vqa(data_samples[0], question=args.question)

    print(f"\nQuestion: {result['question']}")
    print(f"Answer:   {result['answer']}")

    if args.output:
        engine.save_result(result, args.output)


def cmd_gui(args):
    """Launch the Gradio GUI."""
    from src.gui import launch_gui

    config = load_config()
    if args.host:
        config.gui_host = args.host
    if args.port:
        config.gui_port = args.port

    launch_gui(config, share=args.share)


def main():
    parser = argparse.ArgumentParser(
        prog="vlam-alpamayo",
        description="VLAM-Alpamayo: Autonomous driving reasoning with NVIDIA Alpamayo models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    info_parser = subparsers.add_parser("info", help="Show model and dataset information")
    info_parser.set_defaults(func=cmd_info)

    # run
    run_parser = subparsers.add_parser("run", help="Run reasoning inference")
    run_parser.add_argument(
        "-m", "--model",
        choices=list(MODEL_IDS.keys()),
        help="Model to use (default: from config)",
    )
    run_parser.add_argument(
        "-n", "--num-data",
        type=int, default=1,
        help="Number of data samples to process (default: 1)",
    )
    run_parser.add_argument(
        "-s", "--samples",
        type=int,
        help="Number of trajectory samples per data point (default: from config)",
    )
    run_parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated in output/)",
    )
    run_parser.set_defaults(func=cmd_run)

    # vqa
    vqa_parser = subparsers.add_parser("vqa", help="Visual Question Answering (Alpamayo 1.5 only)")
    vqa_parser.add_argument(
        "question",
        help="Question about the driving scene",
    )
    vqa_parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )
    vqa_parser.set_defaults(func=cmd_vqa)

    # gui
    gui_parser = subparsers.add_parser("gui", help="Launch the web-based GUI")
    gui_parser.add_argument("--host", help="Host address (default: from config)")
    gui_parser.add_argument("--port", type=int, help="Port (default: from config)")
    gui_parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    gui_parser.set_defaults(func=cmd_gui)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
