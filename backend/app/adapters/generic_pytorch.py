import time
import os
import sys
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModelAdapter


class GenericPyTorchAdapter(BaseModelAdapter):
    """
    Generic PyTorch Model Adapter for arbitrary user models.
    Supports:
      - Dynamic architecture reconstruction from user-provided Python file (e.g. custom/model.py)
      - Serialized nn.Module checkpoints
      - Flexible input resolutions (e.g. 32x32, 224x224)
      - Custom class vocabularies (e.g. circle, square, triangle, cross)
      - Dynamic hook registration across feature extractors for 3D X-Ray activation tracing
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_def_path: Optional[str] = None,
        class_name: Optional[str] = None,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        classes: Optional[List[str]] = None,
        device: str = "cpu",
        custom_name: str = "Generic PyTorch Model",
        dataset_path: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model_def_path = model_def_path
        self.class_name = class_name
        self.input_shape = input_shape
        self.classes = classes or ["circle", "square", "triangle", "cross"]
        self.custom_name = custom_name
        self.dataset_path = dataset_path
        self.model: Optional[nn.Module] = None
        self.is_loaded = False
        self._layer_activations: Dict[str, torch.Tensor] = {}

        if checkpoint_path:
            self.load(
                checkpoint_path=checkpoint_path,
                model_def_path=model_def_path,
                class_name=class_name,
                device=device,
                classes=classes,
                input_shape=input_shape,
                dataset_path=dataset_path,
            )

    def load(
        self,
        checkpoint_path: str,
        model_def_path: Optional[str] = None,
        class_name: Optional[str] = None,
        device: str = "cpu",
        classes: Optional[List[str]] = None,
        input_shape: Optional[Tuple[int, int, int]] = None,
        dataset_path: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device)
        resolved_ckpt = Path(checkpoint_path).resolve()
        self.checkpoint_path = str(resolved_ckpt)

        if classes:
            self.classes = classes
        if input_shape:
            self.input_shape = input_shape
        if dataset_path:
            self.dataset_path = dataset_path

        if not resolved_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resolved_ckpt}")

        model_inst = None

        # Strategy 1: User provided a python model definition file
        def_path = model_def_path or self.model_def_path
        if def_path and Path(def_path).exists():
            resolved_def = Path(def_path).resolve()
            self.model_def_path = str(resolved_def)
            mod_name = f"custom_model_{resolved_def.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, str(resolved_def))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)

                # Find nn.Module class
                target_cls = None
                if class_name and hasattr(module, class_name):
                    target_cls = getattr(module, class_name)
                else:
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if inspect.isclass(attr) and issubclass(attr, nn.Module) and attr != nn.Module:
                            target_cls = attr
                            self.class_name = attr_name
                            break

                if target_cls:
                    try:
                        # Instantiate with num_classes if supported, else default
                        try:
                            model_inst = target_cls(num_classes=len(self.classes))
                        except TypeError:
                            model_inst = target_cls()
                    except Exception as e:
                        raise ValueError(f"Failed to instantiate model class '{target_cls.__name__}': {e}")

        # Strategy 2: Attempt torch.load
        try:
            checkpoint = torch.load(str(resolved_ckpt), map_location=self.device, weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint file '{resolved_ckpt}': {e}")

        if isinstance(checkpoint, nn.Module):
            model_inst = checkpoint
        elif isinstance(checkpoint, dict):
            state_dict = None
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                model_inst = checkpoint["model"]
            else:
                state_dict = checkpoint

            if model_inst is not None and state_dict is not None:
                # Clean DataParallel prefixes if present
                cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model_inst.load_state_dict(cleaned_sd, strict=False)

        if model_inst is None:
            raise ValueError(
                "Could not construct model instance. If providing a state_dict checkpoint, "
                "please include the corresponding Python model definition file (.py)."
            )

        model_inst.to(self.device)
        model_inst.eval()

        # Validate forward pass with test tensor matching input_shape
        c, h, w = self.input_shape
        dummy_input = torch.zeros((1, c, h, w), device=self.device)
        try:
            with torch.no_grad():
                test_out = model_inst(dummy_input)
                if isinstance(test_out, (tuple, list)):
                    test_out = test_out[0]
                num_out = test_out.shape[1] if test_out.dim() > 1 else test_out.shape[0]
                # If output classes count differs and self.classes is default, adjust classes
                if num_out != len(self.classes) and len(self.classes) == 4:
                    self.classes = [f"Class_{i}" for i in range(num_out)]
        except Exception as err:
            raise ValueError(f"Model failed forward pass validation on shape {self.input_shape}: {err}")

        self.model = model_inst
        self.is_loaded = True
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on intermediate layers to capture representations."""
        self._layer_activations = {}

        def get_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._layer_activations[name] = output.detach()
            return hook

        if not self.model:
            return

        # Check for features sub-module (e.g. NeuralTrailCNN, AlexNet, VGG, MobileNet)
        if hasattr(self.model, "features") and isinstance(self.model.features, nn.Sequential):
            conv_count = 0
            for i, layer in enumerate(self.model.features):
                if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d)):
                    conv_count += 1
                    layer.register_forward_hook(get_hook(f"feat_layer_{i}"))
        else:
            # Fallback: hook named children
            count = 0
            for name, module in self.model.named_children():
                count += 1
                module.register_forward_hook(get_hook(f"stage_{name}"))

    def _ensure_loaded(self) -> None:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Generic PyTorch Model is not loaded.")

    def _to_tensor(self, image: Union[bytes, Image.Image, torch.Tensor]) -> torch.Tensor:
        c, h, w = self.input_shape
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[2:] != (h, w):
                tensor = F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
            return tensor.to(self.device)

        if isinstance(image, bytes):
            import io
            pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Resize to expected input shape
        pil_img = pil_img.resize((w, h), Image.Resampling.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
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

        num_classes = probabilities.size(0)
        k = min(top_k, num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=k)

        predictions: List[Dict[str, Any]] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            idx_int = int(idx)
            label_name = self.classes[idx_int] if idx_int < len(self.classes) else f"Class_{idx_int}"
            predictions.append({
                "index": idx_int,
                "label": label_name,
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

    def predict_and_embed_batch(
        self,
        images: List[Union[bytes, Image.Image, torch.Tensor]],
        top_k: int = 5,
        capture_activations: bool = False,
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

            # Penultimate layer or captured activations for embeddings
            activations = list(self._layer_activations.values())
            if activations:
                feat = activations[-1]
                if feat.dim() > 2:
                    feat = F.adaptive_avg_pool2d(feat, (1, 1))
                embeddings = torch.flatten(feat, 1)
            else:
                embeddings = output

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        num_classes = probabilities.size(1)
        k = min(top_k, num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)

        sample_results = []
        emb_matrix = embeddings.cpu().numpy()

        for i in range(batch_size):
            p_probs = top_probs[i].tolist()
            p_indices = top_indices[i].tolist()

            preds = []
            for prob, idx in zip(p_probs, p_indices):
                idx_int = int(idx)
                label_name = self.classes[idx_int] if idx_int < len(self.classes) else f"Class_{idx_int}"
                preds.append({
                    "index": idx_int,
                    "label": label_name,
                    "probability": float(prob),
                    "confidence_percentage": round(float(prob) * 100.0, 2),
                })

            top_idx = int(p_indices[0])
            top_prob = float(p_probs[0])
            top_label = self.classes[top_idx] if top_idx < len(self.classes) else f"Class_{top_idx}"

            sample_results.append({
                "predicted_index": top_idx,
                "predicted_label": top_label,
                "confidence": top_prob,
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

    def get_embedding(self, image: Union[bytes, Image.Image, torch.Tensor]) -> Dict[str, Any]:
        self._ensure_loaded()
        tensor = self._to_tensor(image)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probabilities = torch.softmax(output, dim=1)[0]
            activations = list(self._layer_activations.values())
            if activations:
                feat = activations[-1]
                if feat.dim() > 2:
                    feat = F.adaptive_avg_pool2d(feat, (1, 1))
                embedding = torch.flatten(feat, 1)[0]
            else:
                embedding = output[0]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        top_prob, top_idx = torch.topk(probabilities, k=1)
        top_index = int(top_idx.item())
        top_label = self.classes[top_index] if top_index < len(self.classes) else f"Class_{top_index}"

        return {
            "model_name": self.custom_name,
            "embedding": embedding.cpu().numpy().tolist(),
            "embedding_dimension": embedding.size(0),
            "predicted_label": top_label,
            "predicted_index": top_index,
            "confidence": float(top_prob.item()),
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
        for name, act in list(self._layer_activations.items())[:6]:
            shape = list(act.shape)
            mean_val = float(act.mean().item())
            std_val = float(act.std().item())
            max_val = float(act.max().item())
            min_val = float(act.min().item())
            sparsity = float((act == 0).float().mean().item())

            if act.dim() >= 4:
                spatial_map = act[0].mean(dim=0).cpu().numpy()
                heatmap_2d = [[round(float(val), 4) for val in row] for row in spatial_map]
            else:
                heatmap_2d = [[round(mean_val, 4)]]

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

    def get_activation_stages(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Extracts activation tensors across canonical network stages for 3D X-Ray network visualization.
        """
        self._ensure_loaded()
        t = self._to_tensor(tensor)

        with torch.no_grad():
            output = self.model(t)
            if isinstance(output, (tuple, list)):
                output = output[0]
            logits = output
            probs = torch.softmax(logits, dim=1)

            # Gather captured hooks
            stages_list = list(self._layer_activations.values())

            # Distribute into early, middle, deep
            n = len(stages_list)
            if n >= 3:
                x_early = stages_list[0]
                x_middle = stages_list[n // 2]
                x_deep = stages_list[-1]
            elif n == 2:
                x_early = stages_list[0]
                x_middle = stages_list[1]
                x_deep = stages_list[1]
            elif n == 1:
                x_early = stages_list[0]
                x_middle = stages_list[0]
                x_deep = stages_list[0]
            else:
                x_early = t
                x_middle = t
                x_deep = t

        stage_info = [
            {"id": "layer-input", "name": "INPUT TENSOR", "stage": "INPUT", "shape": f"{t.shape[1]} × {t.shape[2]} × {t.shape[3]}"},
            {"id": "layer-early", "name": "EARLY STAGE", "stage": "EARLY", "shape": f"{x_early.shape[1]} × {x_early.shape[2]} × {x_early.shape[3]}" if x_early.dim() >= 4 else str(list(x_early.shape))},
            {"id": "layer-middle", "name": "MIDDLE STAGE", "stage": "MIDDLE", "shape": f"{x_middle.shape[1]} × {x_middle.shape[2]} × {x_middle.shape[3]}" if x_middle.dim() >= 4 else str(list(x_middle.shape))},
            {"id": "layer-deep", "name": "DEEP STAGE", "stage": "DEEP", "shape": f"{x_deep.shape[1]} × {x_deep.shape[2]} × {x_deep.shape[3]}" if x_deep.dim() >= 4 else str(list(x_deep.shape))},
            {"id": "layer-output", "name": "OUTPUT HEAD", "stage": "OUTPUT", "shape": f"{len(self.classes)} Classes (Top Logits)"},
        ]

        return {
            "input": t,
            "early": x_early,
            "middle": x_middle,
            "deep": x_deep,
            "output_logits": logits,
            "output_probs": probs,
            "stage_info": stage_info,
        }

    def get_input_shape(self) -> Tuple[int, int, int]:
        return self.input_shape

    def get_classes(self) -> List[str]:
        return self.classes

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
            "architecture": f"Custom PyTorch Model ({self.class_name or 'nn.Module'})",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": len(self.classes),
            "input_resolution": list(self.input_shape),
            "checkpoint_path": self.checkpoint_path,
            "is_loaded": self.is_loaded,
            "layer_names": layer_names,
            "classes": self.classes,
        }
