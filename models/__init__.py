from .attention_pooling import AttentionPoolingModel
from .early_fusion import EarlyFusion
from .fine_grained_fusion import FineGrainedFusion
from .memocmt import MemoCMT

# Словарь для автоматического импорта
MODEL_REGISTRY = {
    "early_fusion": EarlyFusion,
    "attention_pooling": AttentionPoolingModel,
    "fine_grained_fusion": FineGrainedFusion,
    "memocmt": MemoCMT,
}


def get_model(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"model {model_name} not found. aviable models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]
