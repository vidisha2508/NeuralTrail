import time
from typing import Dict, Any, Optional
from PIL import Image

from .perturbations import perturbation_composer, image_to_base64_data_url
from ..adapters.base import BaseModelAdapter


class WhatIfExperimentEngine:
    """
    Executes real-time What-If stress testing experiments:
    original image → original prediction → perturbation → new prediction.
    """

    def __init__(self, adapter: BaseModelAdapter):
        self.adapter = adapter

    def run_experiment(
        self,
        image: Image.Image,
        sample_id: str = "SMP-CANONICAL",
        rotation: float = 0.0,
        noise: float = 0.0,
        blur: float = 0.0,
        crop: float = 0.0,
        brightness: float = 0.0,
        occlusion: float = 0.0,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # Step 1: Original Image Inference
        orig_pred = self.adapter.predict(image, top_k=5)
        orig_label = orig_pred["top_label"]
        orig_idx = int(orig_pred["top_index"])
        orig_conf = float(orig_pred["confidence"])

        # Step 2: Apply Modular Perturbations
        params = {
            "rotation": float(rotation),
            "noise": float(noise),
            "blur": float(blur),
            "crop": float(crop),
            "brightness": float(brightness),
            "occlusion": float(occlusion),
        }
        perturbed_image = perturbation_composer.compose(image, params)

        # Step 3: Perturbed Image Inference
        pert_pred = self.adapter.predict(perturbed_image, top_k=5)
        pert_label = pert_pred["top_label"]
        pert_idx = int(pert_pred["top_index"])
        pert_conf = float(pert_pred["confidence"])

        # Step 4: Compute Differential Metrics
        conf_change = round(pert_conf - orig_conf, 4)
        is_flipped = bool(pert_idx != orig_idx)

        # Check if original class is still present in perturbed top-5
        orig_in_top5 = False
        orig_rank_in_pert = None
        for rank, p in enumerate(pert_pred["predictions"]):
            if p["index"] == orig_idx:
                orig_in_top5 = True
                orig_rank_in_pert = rank + 1
                break

        # Compute Scientific Robustness Score (0 - 100)
        if not is_flipped:
            # Prediction maintained
            conf_retention = pert_conf / max(0.01, orig_conf)
            score = 60.0 + (min(1.0, conf_retention) * 40.0)
            robustness_score = int(round(max(55, min(100, score))))
        else:
            # Prediction flipped
            if orig_in_top5:
                # Retained top-5 presence
                score = 25.0 + (10.0 / (orig_rank_in_pert or 5)) + (pert_conf * 10.0)
                robustness_score = int(round(max(20, min(48, score))))
            else:
                # Catastrophic boundary crossing
                score = max(5, int(round(20.0 - (abs(rotation) * 0.15 + noise * 0.1 + blur * 0.8 + occlusion * 0.15))))
                robustness_score = int(max(5, min(25, score)))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        orig_url = image_to_base64_data_url(image) if include_images else None
        pert_url = image_to_base64_data_url(perturbed_image) if include_images else None

        return {
            "sample_id": sample_id,
            "original_prediction": orig_label,
            "original_prediction_idx": orig_idx,
            "original_confidence": round(orig_conf, 4),
            "original_confidence_percentage": round(orig_conf * 100.0, 2),
            "perturbed_prediction": pert_label,
            "perturbed_prediction_idx": pert_idx,
            "perturbed_confidence": round(pert_conf, 4),
            "perturbed_confidence_percentage": round(pert_conf * 100.0, 2),
            "confidence_change": conf_change,
            "confidence_change_percentage": round(conf_change * 100.0, 2),
            "prediction_flip": is_flipped,
            "robustness_score": robustness_score,
            "parameters_applied": params,
            "original_image_url": orig_url,
            "perturbed_image_url": pert_url,
            "top_original_predictions": orig_pred["predictions"],
            "top_perturbed_predictions": pert_pred["predictions"],
            "latency_ms": round(latency_ms, 2),
        }
