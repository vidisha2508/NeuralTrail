from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Tuple, Optional
from PIL import Image
import torch

class BaseModelAdapter(ABC):
    """
    Abstract Model Adapter interface for Neural Trail.
    Enables pluggable model architectures (ResNet18, MobileNet, and Generic PyTorch models).
    All analysis stages (FIND, TRACE, STRESS, GEMMA, EXPERIMENT) communicate through this common interface.
    """

    @abstractmethod
    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Loads weights from a local .pth / .pt checkpoint."""
        pass

    @abstractmethod
    def predict(self, image: Union[bytes, Image.Image, torch.Tensor], top_k: int = 5) -> Dict[str, Any]:
        """
        Runs inference and returns predicted class, probabilities, confidence score, and latency.
        """
        pass

    @abstractmethod
    def predict_and_embed_batch(
        self,
        images: List[Union[bytes, Image.Image, torch.Tensor]],
        top_k: int = 5,
        capture_activations: bool = False,
    ) -> Dict[str, Any]:
        """
        Batch-level forward pass extracting predictions, embeddings, and activation summaries.
        """
        pass

    @abstractmethod
    def get_embedding(self, image: Union[bytes, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        Extracts the latent representation / embedding vector before the classification head.
        """
        pass

    @abstractmethod
    def get_activations(self, image: Union[bytes, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        Extracts intermediate layer activations and spatial feature summaries.
        """
        pass

    @abstractmethod
    def get_activation_stages(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Extracts activation tensors across canonical network stages for 3D X-Ray network visualization:
        {
            "input": tensor,
            "early": tensor,
            "middle": tensor,
            "deep": tensor,
            "output_logits": tensor,
            "output_probs": tensor,
            "stage_info": list of stage metadata
        }
        """
        pass

    @abstractmethod
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Returns expected input shape (Channels, Height, Width), e.g. (3, 224, 224) or (3, 32, 32)."""
        pass

    @abstractmethod
    def get_classes(self) -> List[str]:
        """Returns list of class names recognized by this model."""
        pass

    @abstractmethod
    def _to_tensor(self, image: Union[bytes, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Converts an input image (bytes, PIL Image, or Tensor) to model input tensor on the active device."""
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Returns model architecture, parameter statistics, layer names, and execution device.
        """
        pass
