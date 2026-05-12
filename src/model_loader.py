"""Model loader for Alpamayo models (v1 and v1.5)."""

import torch
from huggingface_hub import login
from transformers import BitsAndBytesConfig

from src.config import AppConfig

# Model IDs on Hugging Face
MODEL_IDS = {
    "alpamayo-1": "nvidia/Alpamayo-R1-10B",
    "alpamayo-1.5": "nvidia/Alpamayo-1.5-10B",
}

# Model metadata for display
MODEL_INFO = {
    "alpamayo-1": {
        "name": "Alpamayo 1 (R1-10B)",
        "backbone": "Cosmos-Reason",
        "params": "10.5B (8.2B backbone + 2.3B action expert)",
        "features": ["Chain-of-Causation reasoning", "Trajectory prediction"],
        "code_repo": "https://github.com/NVlabs/alpamayo",
    },
    "alpamayo-1.5": {
        "name": "Alpamayo 1.5 (10B)",
        "backbone": "Cosmos-Reason2",
        "params": "10.5B (8.2B backbone + 2.3B action expert)",
        "features": [
            "Chain-of-Causation reasoning",
            "Trajectory prediction",
            "Navigation guidance",
            "Flexible camera counts",
            "Visual Question Answering (VQA)",
            "RL post-trained",
        ],
        "code_repo": "https://github.com/NVlabs/alpamayo1.5",
    },
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def authenticate_hf(token: str) -> None:
    """Authenticate with Hugging Face using the provided token."""
    if not token:
        raise ValueError(
            "Hugging Face token is required. Copy config/.env.sample to config/.env "
            "and set HUGGINGFACE_API_TOKEN, or set HF_TOKEN environment variable."
        )
    login(token=token)


def load_model(config: AppConfig, model_key: str | None = None):
    """Load an Alpamayo model based on the config.

    Args:
        config: Application configuration.
        model_key: Model to load ("alpamayo-1" or "alpamayo-1.5").
                   Uses config.default_model if not specified.

    Returns:
        Loaded model on the configured device.
    """
    model_key = model_key or config.default_model
    if model_key not in MODEL_IDS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODEL_IDS.keys())}")

    authenticate_hf(config.huggingface_token)

    model_id = MODEL_IDS[model_key]
    dtype = DTYPE_MAP.get(config.dtype, torch.bfloat16)
    device = config.device

    # Use device_map="auto" to spread model across multiple GPUs when needed
    num_gpus = torch.cuda.device_count() if device == "cuda" else 0

    # 4-bit quantization to reduce memory footprint (~20GB → ~5GB)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # With 4-bit, model fits on a single GPU — use device 0 to avoid
    # cross-device tensor errors from device_map="auto" splitting
    device_map = {"": 0}

    print(f"Loading {MODEL_INFO[model_key]['name']} from {model_id}...")
    print(f"  Device: {device}, Dtype: {config.dtype}, Attention: {config.attn_implementation}")
    print(f"  Quantization: 4-bit (NF4), single GPU ({num_gpus} available)")

    if model_key == "alpamayo-1":
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

        model = AlpamayoR1.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation=config.attn_implementation,
            device_map=device_map,
            quantization_config=quantization_config,
        )
    else:
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        model = Alpamayo1_5.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation=config.attn_implementation,
            device_map=device_map,
            quantization_config=quantization_config,
        )

    print(f"Model loaded successfully.")
    return model


def get_model_list() -> list[dict]:
    """Return metadata for all available models."""
    return [
        {"key": key, **info}
        for key, info in MODEL_INFO.items()
    ]
