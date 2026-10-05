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


class MobileNetAdapter(BaseModelAdapter):
    """
    Adapter for PyTorch MobileNetV2 vision model.
    Supports .pth checkpoint loading, forward inference, feature embedding extraction,
    and intermediate layer activation analysis.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model: Optional[models.MobileNetV2] = None
        self.is_loaded = False
        self._layer_activations: Dict[str, torch.Tensor] = {}

        if checkpoint_path:
            self.load(checkpoint_path, device)

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        resolved_path = Path(checkpoint_path).resolve()
        self.checkpoint_path = str(resolved_path)

        model = models.mobilenet_v2(weights=None)

        if not resolved_path.exists():
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                default_weights = models.MobileNet_V2_Weights.DEFAULT
                model = models.mobilenet_v2(weights=default_weights)
            except Exception:
                model = models.mobilenet_v2(weights=None)
            torch.save(model.state_dict(), str(resolved_path))

        checkpoint = torch.load(str(resolved_path), map_location=self.device, weights_only=True)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "") if k.startswith("module.") else k
            cleaned_state_dict[key] = v

        try:
            model.load_state_dict(cleaned_state_dict, strict=False)
        except Exception:
            pass

        model.to(self.device)
        model.eval()

        self.model = model
        self.is_loaded = True
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._layer_activations = {}

        def get_hook(name: str):
            def hook(module, input, output):
                self._layer_activations[name] = output.detach()
            return hook

        if self.model and hasattr(self.model, "features"):
            try:
                self.model.features[2].register_forward_hook(get_hook("layer1"))
                self.model.features[7].register_forward_hook(get_hook("layer2"))
                self.model.features[14].register_forward_hook(get_hook("layer3"))
                self.model.features[18].register_forward_hook(get_hook("layer4"))
            except Exception:
                pass

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
            "model_name": "MobileNet",
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
            feat = self.model.features(tensor)
            pool = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            embedding = torch.flatten(pool, 1)[0]
            logits = self.model.classifier(embedding.unsqueeze(0))[0]
            probabilities = torch.softmax(logits, dim=0)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        top_prob, top_idx = torch.topk(probabilities, k=1)
        top_index = int(top_idx.item())

        return {
            "model_name": "MobileNet",
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
            x = batch_tensor
            feat = self.model.features(x)
            pool = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            embeddings = torch.flatten(pool, 1)  # (B, 1280)
            logits = self.model.classifier(embeddings)
            probabilities = torch.softmax(logits, dim=1)

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
            "model_name": "MobileNet",
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

        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            act = self._layer_activations.get(layer_name)
            if act is not None:
                shape = list(act.shape)
                mean_val = float(act.mean().item())
                std_val = float(act.std().item())
                max_val = float(act.max().item())
                min_val = float(act.min().item())
                sparsity = float((act == 0).float().mean().item())

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
            "model_name": "MobileNet",
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
        t = self._to_tensor(tensor)

        with torch.no_grad():
            input_tensor = t.clone()
            # Stage 1: Initial conv + first inverted residual
            x = self.model.features[0:3](t)
            x_early = x

            # Stage 2: Middle bottleneck blocks
            x = self.model.features[3:8](x)
            x_middle = x

            # Stage 3: Deep bottleneck blocks + conv 1x1
            x = self.model.features[8:](x)
            x_deep = x

            # Stage 4: AvgPool + Classifier
            pool = torch.nn.functional.adaptive_avg_pool2d(x_deep, (1, 1))
            flat = torch.flatten(pool, 1)
            logits = self.model.classifier(flat)
            probs = torch.softmax(logits, dim=1)

        stage_info = [
            {"id": "layer-input", "name": "INPUT TENSOR", "stage": "INPUT", "shape": f"{t.shape[1]} × {t.shape[2]} × {t.shape[3]}"},
            {"id": "layer-early", "name": "FEATURES[0..2] (EARLY)", "stage": "EARLY", "shape": f"{x_early.shape[1]} × {x_early.shape[2]} × {x_early.shape[3]}"},
            {"id": "layer-middle", "name": "FEATURES[3..7] (MIDDLE)", "stage": "MIDDLE", "shape": f"{x_middle.shape[1]} × {x_middle.shape[2]} × {x_middle.shape[3]}"},
            {"id": "layer-deep", "name": "FEATURES[8..18] (DEEP)", "stage": "DEEP", "shape": f"{x_deep.shape[1]} × {x_deep.shape[2]} × {x_deep.shape[3]}"},
            {"id": "layer-output", "name": "CLASSIFIER (OUTPUT)", "stage": "OUTPUT", "shape": f"{logits.shape[1]} Classes (Top 36 Logits)"},
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
            "model_name": "MobileNet",
            "architecture": "MobileNetV2 Lightweight Inverted Residual Network",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": 1000,
            "input_resolution": [3, 224, 224],
            "checkpoint_path": self.checkpoint_path,
            "is_loaded": self.is_loaded,
            "layer_names": layer_names,
        }
