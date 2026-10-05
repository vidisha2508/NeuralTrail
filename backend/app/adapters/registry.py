from typing import Dict, Type, List, Optional
from .base import BaseModelAdapter
from .resnet18 import ResNet18Adapter
from .mobilenet import MobileNetAdapter
from .generic_pytorch import GenericPyTorchAdapter


class ModelRegistry:
    """
    Central Model Registry for Neural Trail.
    Decouples analysis pipeline (FIND, TRACE, STRESS, GEMMA, EXPERIMENT) from concrete PyTorch architectures.
    Provides uniform adapter interfaces for:
      - ResNet18 (Torchvision)
      - MobileNet (Torchvision MobileNetV2)
      - GenericPyTorchAdapter (Arbitrary user architectures, e.g. NeuralTrailCNN test model or custom uploads)
    """

    _REGISTRY: Dict[str, Type[BaseModelAdapter]] = {
        "resnet18": ResNet18Adapter,
        "mobilenet": MobileNetAdapter,
        "custom": GenericPyTorchAdapter,
        "generic": GenericPyTorchAdapter,
    }

    @classmethod
    def get_adapter_class(cls, model_type: str) -> Type[BaseModelAdapter]:
        key = model_type.lower()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Unknown model architecture '{model_type}'. Registered architectures: {list(cls._REGISTRY.keys())}"
            )
        return cls._REGISTRY[key]

    @classmethod
    def list_available(cls) -> List[str]:
        return ["resnet18", "mobilenet", "custom"]
