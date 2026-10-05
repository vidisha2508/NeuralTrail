import time
import os
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

from .base import BaseModelAdapter
from ..services.preprocessor import preprocessor
from ..services.labels import get_class_label

class ResNet18Adapter(BaseModelAdapter):
    """
    Adapter for PyTorch ResNet18 vision model.
    Supports .pth checkpoint loading, forward inference, feature embedding extraction,
    and intermediate layer activation analysis.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model: Optional[models.ResNet] = None
        self.is_loaded = False
        self._layer_activations: Dict[str, torch.Tensor] = {}

        if checkpoint_path:
            self.load(checkpoint_path, device)

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        resolved_path = Path(checkpoint_path).resolve()
        self.checkpoint_path = str(resolved_path)

        # Instantiate clean ResNet-18 skeleton
        model = models.resnet18(weights=None)

        if not resolved_path.exists():
            # If checkpoint does not exist yet, download default ImageNet checkpoint
            # and save to the specified path to establish the local .pth asset
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                default_weights = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=default_weights)
            except Exception:
                model = models.resnet18(weights=None)
            torch.save(model.state_dict(), str(resolved_path))

        # Load weights from local .pth file
        checkpoint = torch.load(str(resolved_path), map_location=self.device, weights_only=True)

        # Extract state_dict if wrapped in metadata dict
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

        # Remove prefix like 'module.' if saved with DataParallel
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "") if k.startswith("module.") else k
            cleaned_state_dict[key] = v

        model.load_state_dict(cleaned_state_dict, strict=False)
        model.to(self.device)
        model.eval()

        self.model = model
        self.is_loaded = True
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks to capture intermediate activations for layer analysis."""
        self._layer_activations = {}

        def get_hook(name: str):
            def hook(module, input, output):
                self._layer_activations[name] = output.detach()
            return hook

        if self.model:
            self.model.layer1.register_forward_hook(get_hook("layer1"))
            self.model.layer2.register_forward_hook(get_hook("layer2"))
            self.model.layer3.register_forward_hook(get_hook("layer3"))
            self.model.layer4.register_forward_hook(get_hook("layer4"))

    def _ensure_loaded(self) -> None:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Call load(checkpoint_path) first.")

    def _to_tensor(self, image: Union[bytes, Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            tensor = preprocessor.preprocess(image)
        return tensor.to(self.device)

    def predict(self, image: Union[bytes, Image.Image, torch.Tensor], top_k: int = 5) -> Dict[str, Any]:
        self._ensure_loaded()
        tensor = self._to_tensor(image)

        start_time = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(0)))

        predictions: List[Dict[str, Any]] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            predictions.append({
                "index": int(idx),
                "label": get_class_label(idx),
                "probability": float(prob),
                "confidence_percentage": round(float(prob) * 100.0, 2),
            })

        top_pred = predictions[0]

        return {
            "model_name": "ResNet18",
            "top_label": top_pred["label"],
            "top_index": top_pred["index"],
            "confidence": top_pred["probability"],
            "predictions": predictions,
            "inference_time_ms": round(latency_ms, 2),
            "input_shape": list(tensor.shape),
        }

    def get_embedding(self, image: Union[bytes, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        self._ensure_loaded()
        tensor = self._to_tensor(image)

        start_time = time.perf_counter()
        with torch.no_grad():
            # Pass through convolutional backbone up to avgpool
            x = self.model.conv1(tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            embedding_tensor = torch.flatten(x, 1)[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        norm = float(torch.norm(embedding_tensor).item())
        embedding_list = [round(float(v), 6) for v in embedding_tensor.tolist()]

        return {
            "model_name": "ResNet18",
            "dimension": len(embedding_list),
            "embedding": embedding_list,
            "norm": round(norm, 4),
            "inference_time_ms": round(latency_ms, 2),
        }

    def predict_and_embed_batch(
        self,
        images: Union[List[Union[bytes, Image.Image, torch.Tensor]], torch.Tensor],
        top_k: int = 5,
        capture_activations: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs batched forward pass through ResNet-18.
        Simultaneously extracts logits, top predictions, 512-dim latent embeddings,
        and intermediate layer activations in a single vectorized pass.
        """
        self._ensure_loaded()

        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                batch_tensor = images.unsqueeze(0)
            else:
                batch_tensor = images
        else:
            tensors = [self._to_tensor(img).squeeze(0) for img in images]
            batch_tensor = torch.stack(tensors)

        batch_tensor = batch_tensor.to(self.device)
        batch_size = batch_tensor.shape[0]

        start_time = time.perf_counter()
        with torch.no_grad():
            x = self.model.conv1(batch_tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x1 = self.model.layer1(x)
            x2 = self.model.layer2(x1)
            x3 = self.model.layer3(x2)
            x4 = self.model.layer4(x3)

            pool = self.model.avgpool(x4)
            embeddings = torch.flatten(pool, 1)  # (B, 512)
            logits = self.model.fc(embeddings)   # (B, 1000)
            probabilities = torch.softmax(logits, dim=1)  # (B, 1000)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)), dim=1)

        sample_results = []
        emb_matrix = embeddings.cpu().numpy()

        for i in range(batch_size):
            p_probs = top_probs[i].tolist()
            p_indices = top_indices[i].tolist()

            preds = []
            for prob, idx in zip(p_probs, p_indices):
                preds.append({
                    "index": int(idx),
                    "label": get_class_label(idx),
                    "probability": float(prob),
                    "confidence_percentage": round(float(prob) * 100.0, 2),
                })

            top_idx = p_indices[0]
            top_prob = p_probs[0]

            sample_results.append({
                "predicted_index": int(top_idx),
                "predicted_label": get_class_label(top_idx),
                "confidence": float(top_prob),
                "predictions": preds,
                "embedding": emb_matrix[i].tolist(),
            })

        activations_summary = {}
        if capture_activations:
            for l_name, l_tensor in [("layer1", x1), ("layer2", x2), ("layer3", x3), ("layer4", x4)]:
                activations_summary[l_name] = {
                    "shape": list(l_tensor.shape),
                    "mean": round(float(l_tensor.mean().item()), 4),
                    "std": round(float(l_tensor.std().item()), 4),
                    "sparsity": round(float((l_tensor == 0).float().mean().item()), 4),
                }

        return {
            "model_name": "ResNet18",
            "batch_size": batch_size,
            "results": sample_results,
            "embeddings": emb_matrix,
            "activations_summary": activations_summary,
            "inference_time_ms": round(latency_ms, 2),
        }

    def get_activations(self, image: Union[bytes, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        self._ensure_loaded()
        tensor = self._to_tensor(image)

        start_time = time.perf_counter()
        with torch.no_grad():
            _ = self.model(tensor)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        layer_summaries: List[Dict[str, Any]] = []

        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            act = self._layer_activations.get(layer_name)
            if act is not None:
                shape = list(act.shape)
                mean_val = float(act.mean().item())
                std_val = float(act.std().item())
                max_val = float(act.max().item())
                min_val = float(act.min().item())
                sparsity = float((act == 0).float().mean().item())

                # Generate a 2D spatial heatmap across channels (B, C, H, W) -> (H, W)
                spatial_map = act[0].mean(dim=0).cpu().numpy()
                heatmap_2d = [[round(float(val), 4) for val in row] for row in spatial_map]

                layer_summaries.append({
                    "layer_name": layer_name,
                    "shape": shape,
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "max": round(max_val, 4),
                    "min": round(min_val, 4),
                    "sparsity": round(sparsity, 4),
                    "heatmap_2d": heatmap_2d,
                })

        return {
            "model_name": "ResNet18",
            "layers": layer_summaries,
            "inference_time_ms": round(latency_ms, 2),
        }

    def get_input_shape(self) -> Tuple[int, int, int]:
        return (3, 224, 224)

    def get_classes(self) -> List[str]:
        from ..services.labels import IMAGENET_1K_LABELS
        return IMAGENET_1K_LABELS

    def get_activation_stages(self, tensor: torch.Tensor) -> Dict[str, Any]:
        self._ensure_loaded()
        with torch.no_grad():
            t = tensor.to(self.device)
            input_tensor = t.clone()
            x = self.model.conv1(t)
            x = self.model.bn1(x)
            x_early = self.model.relu(x)
            x = self.model.maxpool(x_early)

            x_l1 = self.model.layer1(x)
            x_middle = self.model.layer2(x_l1)

            x_l3 = self.model.layer3(x_middle)
            x_deep = self.model.layer4(x_l3)

            pool = self.model.avgpool(x_deep)
            flat = torch.flatten(pool, 1)
            logits = self.model.fc(flat)
            probs = torch.softmax(logits, dim=1)

        stage_info = [
            {"id": "layer-input", "name": "INPUT TENSOR", "stage": "INPUT", "shape": f"{tensor.shape[1]} × {tensor.shape[2]} × {tensor.shape[3]}"},
            {"id": "layer-early", "name": "CONV1 + RELU (EARLY)", "stage": "EARLY", "shape": f"{x_early.shape[1]} × {x_early.shape[2]} × {x_early.shape[3]}"},
            {"id": "layer-middle", "name": "LAYER2.RES (MIDDLE)", "stage": "MIDDLE", "shape": f"{x_middle.shape[1]} × {x_middle.shape[2]} × {x_middle.shape[3]}"},
            {"id": "layer-deep", "name": "LAYER4.RES (DEEP)", "stage": "DEEP", "shape": f"{x_deep.shape[1]} × {x_deep.shape[2]} × {x_deep.shape[3]}"},
            {"id": "layer-output", "name": "FC + SOFTMAX (OUTPUT)", "stage": "OUTPUT", "shape": f"{logits.shape[1]} Classes (Top 36 Logits)"},
        ]

        return {
            "input": input_tensor,
            "early": x_early,
            "middle": x_middle,
            "deep": x_deep,
            "output_logits": logits,
            "output_probs": probs,
            "stage_info": stage_info,
        }

    def metadata(self) -> Dict[str, Any]:
        total_params = 0
        trainable_params = 0
        layer_names = []

        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            layer_names = [name for name, _ in self.model.named_children()]

        return {
            "model_name": "ResNet18",
            "architecture": "Deep Residual Convolutional Network (18 Layers)",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": 1000,
            "input_resolution": [3, 224, 224],
            "checkpoint_path": self.checkpoint_path,
            "is_loaded": self.is_loaded,
            "layer_names": layer_names,
        }

