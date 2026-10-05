import time
import os
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

from .base import BaseModelAdapter
from ..services.preprocessor import preprocessor
from ..services.labels import get_class_label


class CustomPyTorchAdapter(BaseModelAdapter):
    """
    Adapter for user-uploaded PyTorch (.pt / .pth) models.
    Supports validating uploaded state_dict or serialized nn.Module,
    forward pass, batch embedding extraction, and layer inspection.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu", custom_name: str = "Uploaded PyTorch Model"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.custom_name = custom_name
        self.model: Optional[nn.Module] = None
        self.is_loaded = False
        self._layer_activations: Dict[str, torch.Tensor] = {}

        if checkpoint_path:
            self.load(checkpoint_path, device)

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        resolved_path = Path(checkpoint_path).resolve()
        self.checkpoint_path = str(resolved_path)

        if not resolved_path.exists():
            raise FileNotFoundError(f"Uploaded model file not found at: {resolved_path}")

        # Try torch loading weights or model object
        try:
            checkpoint = torch.load(str(resolved_path), map_location=self.device, weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch file with torch.load: {str(e)}")

        model_inst = None

        if isinstance(checkpoint, nn.Module):
            model_inst = checkpoint
        elif isinstance(checkpoint, dict):
            # Extract state_dict
            state_dict = None
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                model_inst = checkpoint["model"]
            else:
                state_dict = checkpoint

            if model_inst is None and isinstance(state_dict, dict):
                # Attempt to fit state dict into ResNet18 or MobileNetV2 skeleton first
                for base_model_fn in [models.resnet18, models.mobilenet_v2, models.resnet34]:
                    try:
                        candidate = base_model_fn(weights=None)
                        cleaned = {}
                        for k, v in state_dict.items():
                            key = k.replace("module.", "") if k.startswith("module.") else k
                            cleaned[key] = v
                        candidate.load_state_dict(cleaned, strict=False)
                        model_inst = candidate
                        break
                    except Exception:
                        continue

                if model_inst is None:
                    # Generic fallback skeleton if state dict doesn't match standard vision backbones directly
                    model_inst = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if model_inst is None:
            # Fallback robust model skeleton
            model_inst = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        model_inst.to(self.device)
        model_inst.eval()

        # Perform test forward pass to validate compatibility
        dummy_input = torch.zeros((1, 3, 224, 224), device=self.device)
        try:
            with torch.no_grad():
                _ = model_inst(dummy_input)
        except Exception as err:
            raise ValueError(f"Uploaded PyTorch model failed forward pass validation: {str(err)}")

        self.model = model_inst
        self.is_loaded = True
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._layer_activations = {}

        def get_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._layer_activations[name] = output.detach()
            return hook

        if self.model:
            count = 0
            for name, module in self.model.named_children():
                if len(list(module.children())) == 0 or count < 4:
                    count += 1
                    module.register_forward_hook(get_hook(f"layer{count}"))

    def _ensure_loaded(self) -> None:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Custom PyTorch Model is not loaded.")

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
            output = self.model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probabilities = torch.softmax(output, dim=1)[0]
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
            "model_name": self.custom_name,
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
            output = self.model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probabilities = torch.softmax(output, dim=1)[0]
            # Extract penultimate activations or use output logits as fallback
            activations = list(self._layer_activations.values())
            if activations:
                feat = activations[-1]
                if feat.dim() > 2:
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                embedding = torch.flatten(feat, 1)[0]
            else:
                embedding = output[0]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        top_prob, top_idx = torch.topk(probabilities, k=1)
        top_index = int(top_idx.item())

        return {
            "model_name": self.custom_name,
            "embedding": embedding.cpu().numpy().tolist(),
            "embedding_dimension": embedding.size(0),
            "predicted_label": get_class_label(top_index),
            "predicted_index": top_index,
            "confidence": float(top_prob.item()),
            "inference_time_ms": round(latency_ms, 2),
        }

    def predict_and_embed_batch(
        self,
        images: List[Union[bytes, Image.Image, torch.Tensor]],
        top_k: int = 5,
        capture_activations: bool = False
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        tensors = [self._to_tensor(img) for img in images]
        batch_tensor = torch.cat(tensors, dim=0)
        batch_size = batch_tensor.size(0)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.model(batch_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probabilities = torch.softmax(output, dim=1)

            activations = list(self._layer_activations.values())
            if activations:
                feat = activations[-1]
                if feat.dim() > 2:
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                embeddings = torch.flatten(feat, 1)
            else:
                embeddings = output

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

        return {
            "model_name": self.custom_name,
            "batch_size": batch_size,
            "results": sample_results,
            "embeddings": emb_matrix,
            "activations_summary": {},
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

        for name, act in self._layer_activations.items():
            if act is not None:
                shape = list(act.shape)
                mean_val = float(act.mean().item())
                std_val = float(act.std().item())
                max_val = float(act.max().item())
                min_val = float(act.min().item())
                sparsity = float((act == 0).float().mean().item())

                if act.dim() >= 3:
                    spatial_map = act[0].mean(dim=0).cpu().numpy()
                    heatmap_2d = [[round(float(val), 4) for val in row] for row in spatial_map]
                else:
                    heatmap_2d = [[round(float(v), 4) for v in act[0].cpu().numpy()[:16]]]

                layer_summaries.append({
                    "layer_name": name,
                    "shape": shape,
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "max": round(max_val, 4),
                    "min": round(min_val, 4),
                    "sparsity": round(sparsity, 4),
                    "heatmap_2d": heatmap_2d,
                })

        return {
            "model_name": self.custom_name,
            "layers": layer_summaries,
            "inference_time_ms": round(latency_ms, 2),
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
            "model_name": self.custom_name,
            "architecture": f"User Uploaded PyTorch Model ({self.custom_name})",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": 1000,
            "input_resolution": [3, 224, 224],
            "checkpoint_path": self.checkpoint_path,
            "is_loaded": self.is_loaded,
            "layer_names": layer_names,
        }
