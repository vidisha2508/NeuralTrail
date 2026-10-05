import time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from .reducers import get_reducer
from .blind_spots import BlindSpotDetector
from .metrics import MetricsCalculator
from ..datasets.benchmark import BenchmarkDatasetGenerator, BenchmarkSample
from ..datasets.loader import DatasetLoader
from ..adapters.base import BaseModelAdapter


class AnalysisPipeline:
    """
    End-to-End Neural Trail Analysis Pipeline:
    dataset → BaseModelAdapter → predictions + embeddings + activations → analysis.
    Works transparently with ResNet18, MobileNet, or GenericPyTorchAdapter.
    """

    def __init__(self, adapter: BaseModelAdapter):
        self.adapter = adapter
        self.detector = BlindSpotDetector(random_state=42)

    def _tensor_to_thumbnail_grid(self, pil_img: Image.Image) -> List[List[float]]:
        # Downsample to 6x6 grid of brightness values [0.0 - 1.0]
        small = pil_img.resize((6, 6), Image.Resampling.BILINEAR).convert("L")
        arr = np.array(small, dtype=np.float32) / 255.0
        return [[round(float(v), 2) for v in row] for row in arr]

    def _get_dataset_samples(
        self,
        dataset_source: str = "benchmark",
        sample_count: int = 100,
        custom_dir: Optional[str] = None,
    ) -> Tuple[List[BenchmarkSample], str]:
        if dataset_source == "custom" and custom_dir:
            dataset_samples = DatasetLoader.load_from_directory(custom_dir, max_samples=sample_count)
            return dataset_samples, f"Custom Dataset ({custom_dir})"

        # Check if active adapter is using ShapeDataset or custom dataset
        classes = self.adapter.get_classes()
        dataset_path = getattr(self.adapter, "dataset_path", None)
        is_shape_dataset = (
            classes == ["circle", "square", "triangle", "cross"]
            or (dataset_path and "dataset.py" in dataset_path)
            or self.adapter.get_input_shape() == (3, 32, 32)
        )

        if is_shape_dataset:
            from pathlib import Path
            import sys
            import importlib.util

            ds_path = dataset_path or str(Path(__file__).resolve().parents[3] / "custom" / "dataset.py")
            if Path(ds_path).exists():
                spec = importlib.util.spec_from_file_location("custom_dataset_mod", ds_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    shape_ds = mod.ShapeDataset(size=max(sample_count, 100), seed=42)
                    samples = []
                    perturbation_types = ["clean", "clean", "rotation", "noise", "blur", "occlusion"]

                    for idx in range(len(shape_ds)):
                        tensor, label = shape_ds[idx]
                        # tensor is (3, 32, 32) in [0, 1]
                        arr = (tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        pil_img = Image.fromarray(arr)

                        ptype = perturbation_types[idx % len(perturbation_types)]
                        # Apply stress perturbation to some evaluation samples to validate failure clusters
                        if ptype == "rotation":
                            pil_img = pil_img.rotate(45, resample=Image.Resampling.BILINEAR)
                        elif ptype == "blur":
                            from PIL import ImageFilter
                            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.5))
                        elif ptype == "occlusion":
                            draw_pixels = pil_img.load()
                            for y in range(10, 22):
                                for x in range(10, 22):
                                    draw_pixels[x, y] = (0, 0, 0)

                        grid = self._tensor_to_thumbnail_grid(pil_img)
                        class_name = classes[label] if label < len(classes) else f"class_{label}"

                        samples.append(
                            BenchmarkSample(
                                id=f"SMP-{idx+1:03d}",
                                image=pil_img,
                                true_class_idx=int(label),
                                true_class_name=class_name,
                                perturbation=ptype,
                                features={"rotation": 45.0 if ptype == "rotation" else 0.0, "noise": 0.2 if ptype == "noise" else 0.0, "blur": 1.5 if ptype == "blur" else 0.0},
                                thumbnail_grid=grid,
                            )
                        )

                    return samples[:sample_count], "ShapeDataset (4 Classes: circle, square, triangle, cross)"

        # Default: Benchmark Dataset Generator
        count_per_class = max(2, sample_count // 10)
        generator = BenchmarkDatasetGenerator(seed=42)
        dataset_samples = generator.generate(count_per_class=count_per_class)
        dataset_name = "ImageNet-1K Benchmark Suite"
        return dataset_samples, dataset_name

    def run(
        self,
        dataset_source: str = "benchmark",
        sample_count: int = 100,
        reducer_type: str = "pca",
        k_clusters: Optional[int] = None,
        custom_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # Step 1: Ingest Dataset
        dataset_samples, dataset_name = self._get_dataset_samples(
            dataset_source=dataset_source,
            sample_count=sample_count,
            custom_dir=custom_dir,
        )

        if not dataset_samples:
            raise ValueError("No samples available to analyze.")

        # Step 2: Vectorized Batch Inference (predictions + embeddings + activations)
        batch_images = [s.image for s in dataset_samples]
        batch_size = 32
        sample_results = []
        embeddings_list = []
        all_activations = {}

        for i in range(0, len(batch_images), batch_size):
            chunk = batch_images[i : i + batch_size]
            is_last_chunk = (i + batch_size) >= len(batch_images)
            res = self.adapter.predict_and_embed_batch(chunk, top_k=5, capture_activations=is_last_chunk)

            sample_results.extend(res["results"])
            embeddings_list.append(res["embeddings"])
            if res.get("activations_summary"):
                all_activations = res["activations_summary"]

        embeddings_matrix = np.vstack(embeddings_list)

        # Step 3: Dimensionality Reduction (PCA / UMAP / t-SNE)
        reducer = get_reducer(reducer_type)
        coords_2d = reducer.fit_transform(embeddings_matrix)

        # Step 4: Track Sample Level Diagnostics
        tracked_samples: List[Dict[str, Any]] = []
        for idx, (bench_s, res_s) in enumerate(zip(dataset_samples, sample_results)):
            pred_idx = res_s["predicted_index"]
            pred_label = res_s["predicted_label"]
            confidence = float(res_s["confidence"])
            true_idx = bench_s.true_class_idx
            true_name = bench_s.true_class_name

            # Check if prediction is correct
            is_correct = (pred_idx == true_idx)

            # Stability classification
            if not is_correct:
                stability = "failure"
            elif confidence < 0.60:
                stability = "uncertain"
            else:
                stability = "stable"

            tracked_samples.append({
                "id": bench_s.id,
                "x": float(coords_2d[idx, 0]),
                "y": float(coords_2d[idx, 1]),
                "trueClass": true_idx,
                "true_class_idx": true_idx,
                "trueClassName": true_name,
                "true_class_name": true_name,
                "predictedClass": pred_idx,
                "predicted_class_idx": pred_idx,
                "predictedClassName": pred_label,
                "predicted_label": pred_label,
                "confidence": round(confidence, 4),
                "correct": is_correct,
                "stability": stability,
                "perturbation": bench_s.perturbation,
                "sensitivity": bench_s.perturbation.replace("_", " ").title(),
                "thumbnailGrid": bench_s.thumbnail_grid,
                "features": bench_s.features,
                "top_predictions": res_s.get("predictions", []),
                "embedding": res_s["embedding"][:16] if isinstance(res_s["embedding"], list) else [],
            })

        # Step 5: Blind Spot Detection & High-Failure Density Clustering
        blind_spots, cluster_labels = self.detector.detect(
            samples=tracked_samples,
            coords_2d=coords_2d,
            embeddings=embeddings_matrix,
            k_clusters=k_clusters,
        )

        # Assign cluster ID to every sample
        for s, c_idx in zip(tracked_samples, cluster_labels):
            s["clusterId"] = f"cluster-{c_idx}"

        # Step 6: Model Diagnostic Metrics
        model_meta = self.adapter.metadata()
        metrics = MetricsCalculator.calculate(
            samples=tracked_samples,
            blind_spots=blind_spots,
            model_name=model_meta["model_name"],
            architecture=model_meta["architecture"],
            dataset_name=dataset_name,
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "COMPLETE",
            "dataset_name": dataset_name,
            "total_samples": len(tracked_samples),
            "reducer": reducer.name,
            "metrics": metrics,
            "blind_spots": blind_spots,
            "samples": tracked_samples,
            "sample_images": {s.id: s.image for s in dataset_samples},
            "activations": all_activations,
            "latency_ms": round(total_latency_ms, 2),
            "timestamp": time.time(),
        }
