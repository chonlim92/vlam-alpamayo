"""Configuration loader for VLAM-Alpamayo."""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class AppConfig:
    huggingface_token: str
    default_model: str
    device: str
    dtype: str
    attn_implementation: str
    num_traj_samples: int
    output_dir: str
    gui_host: str
    gui_port: int


def load_config() -> AppConfig:
    """Load configuration from config/.env file and environment variables.

    Looks for config/.env (copy from config/.env.sample).
    Environment variables override .env values.
    """
    project_root = Path(__file__).parent.parent
    env_path = project_root / "config" / ".env"

    if env_path.exists():
        load_dotenv(env_path)

    token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    if not token or token in ("your_token_here", "<your-HF-Token>"):
        token = os.getenv("HF_TOKEN", "")

    return AppConfig(
        huggingface_token=token,
        default_model=os.getenv("DEFAULT_MODEL", "alpamayo-1.5"),
        device=os.getenv("DEVICE", "cuda"),
        dtype=os.getenv("DTYPE", "bfloat16"),
        attn_implementation=os.getenv("ATTN_IMPLEMENTATION", "eager"),
        num_traj_samples=int(os.getenv("NUM_TRAJ_SAMPLES", "1")),
        output_dir=os.getenv("OUTPUT_DIR", "output"),
        gui_host=os.getenv("GUI_HOST", "127.0.0.1"),
        gui_port=int(os.getenv("GUI_PORT", "7860")),
    )
