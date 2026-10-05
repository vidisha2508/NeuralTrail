import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from ..adapters.base import BaseModelAdapter


class ActivationPathwayEngine:
    """
    Extracts intermediate layer activations across canonical stages:
    INPUT → EARLY → MIDDLE → DEEP → OUTPUT.
    Enables comparative activation analysis between successful canonical samples
    and failed / perturbed samples across ANY model loaded through BaseModelAdapter.
    """

    DEFAULT_STAGES = [
        {"id": "layer-input", "name": "INPUT TENSOR", "stage": "INPUT", "shape": "3 × 224 × 224"},
        {"id": "layer-early", "name": "EARLY STAGE", "stage": "EARLY", "shape": "Early Feature Map"},
        {"id": "layer-middle", "name": "MIDDLE STAGE", "stage": "MIDDLE", "shape": "Middle Representation"},
        {"id": "layer-deep", "name": "DEEP STAGE", "stage": "DEEP", "shape": "Penultimate Latent"},
        {"id": "layer-output", "name": "OUTPUT HEAD", "stage": "OUTPUT", "shape": "Output Logits"},
    ]

    def __init__(self, adapter: BaseModelAdapter):
        self.adapter = adapter

    def _extract_stages_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Delegates to adapter's generic get_activation_stages implementation.
        """
        self.adapter._ensure_loaded()
        return self.adapter.get_activation_stages(tensor)

    def _tensor_to_6x6_map(self, t: torch.Tensor) -> List[List[float]]:
        """
        Takes tensor (1, C, H, W) or (C, H, W) or 2D tensor and computes spatial 6x6 feature map
        normalized to [0.0, 1.0].
        """
        if t.dim() == 2:
            # Flattened feature vector, pad or pool into 6x6
            vals = t[0].cpu().numpy()
            if len(vals) >= 36:
                arr = vals[:36].reshape((6, 6))
            else:
                padded = np.zeros(36, dtype=np.float32)
                padded[:len(vals)] = vals
                arr = padded.reshape((6, 6))
            p_min, p_max = float(arr.min()), float(arr.max())
            diff = p_max - p_min
            norm = (arr - p_min) / diff if diff > 1e-6 else np.zeros((6, 6), dtype=np.float32)
            return [[round(float(v), 2) for v in row] for row in norm]

        if t.dim() == 3:
            t = t.unsqueeze(0)

        mean_c = t.mean(dim=1, keepdim=True)
        pooled = F.adaptive_avg_pool2d(mean_c, (6, 6))[0, 0].cpu().numpy()
        p_min = float(pooled.min())
        p_max = float(pooled.max())
        diff = p_max - p_min
        if diff > 1e-6:
            norm = (pooled - p_min) / diff
        else:
            norm = np.zeros((6, 6), dtype=np.float32)
        return [[round(float(v), 2) for v in row] for row in norm]

    def _logits_to_6x6_map(self, probs: torch.Tensor) -> List[List[float]]:
        """
        Takes probabilities (1, N) and formats top logits/classes into a 6x6 visual matrix.
        """
        k = min(36, probs.size(1))
        top_vals, _ = torch.topk(probs[0], k=k)
        top_arr = top_vals.cpu().numpy()
        padded = np.zeros(36, dtype=np.float32)
        padded[:k] = top_arr

        p_min = float(padded.min())
        p_max = float(padded.max())
        diff = p_max - p_min
        scaled = (padded - p_min) / diff if diff > 1e-6 else padded
        mat = scaled.reshape((6, 6))
        return [[round(float(v), 2) for v in row] for row in mat]

    def _compute_stats(self, t: torch.Tensor) -> Tuple[float, float]:
        """Returns (activation_norm, dead_neuron_ratio)."""
        norm = float(torch.norm(t).item())
        dead_ratio = float((t == 0).float().mean().item())
        return round(norm, 2), round(dead_ratio, 2)

    def _compute_anomaly_score(self, map_a: List[List[float]], map_b: List[List[float]]) -> float:
        """Computes representation divergence score between two 6x6 feature maps in [0.0, 1.0]."""
        a = np.array(map_a, dtype=np.float32).flatten()
        b = np.array(map_b, dtype=np.float32).flatten()

        mae = float(np.mean(np.abs(a - b)))
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 1e-6 and norm_b > 1e-6:
            cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
            cos_dist = max(0.0, min(1.0, 1.0 - cos_sim))
        else:
            cos_dist = mae

        score = 0.5 * mae + 0.5 * cos_dist
        return round(max(0.02, min(0.98, score)), 2)

    def compare_pathways(
        self,
        img_a: Any,
        img_b: Any,
        sample_meta_a: Dict[str, Any],
        sample_meta_b: Dict[str, Any],
        selected_layer_id: str = "layer-middle",
    ) -> Dict[str, Any]:
        """
        Runs real PyTorch forward pass on both samples, extracting 5-stage activations
        and comparing representation divergence.
        """
        tensor_a = self.adapter._to_tensor(img_a)
        tensor_b = self.adapter._to_tensor(img_b)

        stages_a = self._extract_stages_tensor(tensor_a)
        stages_b = self._extract_stages_tensor(tensor_b)

        stage_meta_list = stages_a.get("stage_info", self.DEFAULT_STAGES)

        mapping = [
            ("layer-input", stages_a["input"], stages_b["input"], "input"),
            ("layer-early", stages_a["early"], stages_b["early"], "conv"),
            ("layer-middle", stages_a["middle"], stages_b["middle"], "conv"),
            ("layer-deep", stages_a["deep"], stages_b["deep"], "conv"),
            ("layer-output", stages_a["output_probs"], stages_b["output_probs"], "output"),
        ]

        layers_data: List[Dict[str, Any]] = []

        for s_info, (layer_id, t_a, t_b, kind) in zip(stage_meta_list, mapping):
            if kind == "output":
                map_a = self._logits_to_6x6_map(t_a)
                map_b = self._logits_to_6x6_map(t_b)
                norm_a = 1.0
                dead_a = 0.0
                norm_b = 1.0
                dead_b = 0.0
            else:
                map_a = self._tensor_to_6x6_map(t_a)
                map_b = self._tensor_to_6x6_map(t_b)
                norm_a, dead_a = self._compute_stats(t_a)
                norm_b, dead_b = self._compute_stats(t_b)

            anomaly = self._compute_anomaly_score(map_a, map_b)

            diff_map = [
                [round(abs(va - vb), 2) for va, vb in zip(row_a, row_b)]
                for row_a, row_b in zip(map_a, map_b)
            ]

            layers_data.append({
                "id": s_info.get("id", layer_id),
                "name": s_info.get("name", layer_id.upper()),
                "stage": s_info.get("stage", "CONV"),
                "tensorShape": s_info.get("shape", str(list(t_a.shape))),
                "activationNorm": norm_a,
                "deadNeuronRatio": dead_a,
                "activationNormPerturbed": norm_b,
                "deadNeuronRatioPerturbed": dead_b,
                "anomalyScore": anomaly,
                "normalFeatureMap": map_a,
                "perturbedFeatureMap": map_b,
                "differenceMap": diff_map,
            })

        return {
            "feature_name": "Activation Pathway Visualization",
            "selected_layer_id": selected_layer_id,
            "layers": layers_data,
            "sample_a": sample_meta_a,
            "sample_b": sample_meta_b,
            "causality_disclaimer": (
                "Correlational Pathway Note: Feature map divergence highlights localized representation shifts "
                "across layers under perturbation, but does not prove causal attribution."
            ),
        }
