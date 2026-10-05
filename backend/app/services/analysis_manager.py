import logging
import threading
import time
from typing import Dict, Any, List, Optional
from ..adapters.base import BaseModelAdapter
from ..analysis.pipeline import AnalysisPipeline
from ..analysis.xray import ActivationPathwayEngine
from ..analysis.whatif import WhatIfExperimentEngine
from ..datasets.benchmark import BenchmarkDatasetGenerator

logger = logging.getLogger("neural-trail-backend")


class AnalysisManager:
    """
    Manages active investigation state, evaluation executions, and session event telemetry.
    Strictly maintains ONE active investigation state across all tabs:
    MODEL → EVALUATE → FIND → TRACE → STRESS → GEMMA → EXPERIMENT → IMPROVE → REPLAY
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._is_running = False
        self._stage = "IDLE"
        self._progress = 0
        self._last_result: Optional[Dict[str, Any]] = None
        self._error_message: Optional[str] = None
        self._last_run_time: Optional[float] = None
        self._active_sample_id: Optional[str] = None
        self._active_cluster_id: Optional[str] = None
        self._investigation_events: List[Dict[str, Any]] = []

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def progress(self) -> int:
        return self._progress

    def reset(self) -> None:
        """Completely clears the investigation state when switching or unloading models."""
        with self._lock:
            self._is_running = False
            self._stage = "IDLE"
            self._progress = 0
            self._last_result = None
            self._error_message = None
            self._last_run_time = None
            self._active_sample_id = None
            self._active_cluster_id = None
            self._investigation_events = []
        logger.info("[AnalysisManager] Investigation state reset.")

    def record_event(
        self,
        code: str,
        title: str,
        short: str,
        event_type: str,
        desc: str,
        telemetry: str,
        accent: str = "cyan",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records an actual empirical event in the investigation timeline."""
        with self._lock:
            evt = {
                "id": f"EVT-{len(self._investigation_events)+1:02d}",
                "code": f"{len(self._investigation_events)+1:02d}",
                "title": title,
                "short": short,
                "type": event_type,
                "desc": desc,
                "telemetry": telemetry,
                "accent": accent,
                "timestamp": time.time(),
                "time_str": time.strftime("%H:%M:%S"),
                "details": details or {},
            }
            self._investigation_events.append(evt)

    def get_replay_events(self) -> List[Dict[str, Any]]:
        """Returns actual investigation events recorded during the current session."""
        with self._lock:
            return list(self._investigation_events)

    def set_active_sample(self, sample_id: str) -> None:
        with self._lock:
            self._active_sample_id = sample_id
        if self._last_result:
            s = next((item for item in self._last_result["samples"] if item["id"] == sample_id), None)
            if s:
                status_str = "Accurate" if s.get("correct") else f"Misclassified as {s.get('predictedClassName')}"
                self.record_event(
                    code="03",
                    title=f"Sample Inspected: #{s['id']}",
                    short="Inspected",
                    event_type="SAMPLE_SELECTED",
                    desc=f"Target sample selected for cross-stage activation tracing. Ground truth: {s.get('trueClassName')}. Status: {status_str}.",
                    telemetry=f"Confidence: {s.get('confidence', 0.0) * 100:.1f}% | Manifold: ({s.get('x', 0):.2f}, {s.get('y', 0):.2f})",
                    accent="green" if s.get("correct") else "magenta",
                )

    def set_active_cluster(self, cluster_id: str) -> None:
        with self._lock:
            self._active_cluster_id = cluster_id

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self._is_running,
            "stage": self._stage,
            "progress": self._progress,
            "has_results": self._last_result is not None,
            "total_samples": self._last_result["total_samples"] if self._last_result else 0,
            "blind_spots_count": len(self._last_result["blind_spots"]) if self._last_result else 0,
            "last_run_time": self._last_run_time,
            "error": self._error_message,
            "active_sample_id": self._active_sample_id,
            "active_cluster_id": self._active_cluster_id,
            "events_count": len(self._investigation_events),
        }

    def start_analysis(
        self,
        dataset_source: str = "benchmark",
        sample_count: int = 100,
        reducer_type: str = "pca",
        k_clusters: Optional[int] = None,
        custom_dir: Optional[str] = None,
        async_run: bool = False,
    ) -> Dict[str, Any]:
        from .model_loader import model_manager
        if not model_manager.is_loaded:
            raise RuntimeError("No model is currently loaded. Load a model before running analysis.")

        with self._lock:
            if self._is_running:
                return {
                    "success": False,
                    "message": "Analysis is already in progress.",
                    "status": self.get_status(),
                }
            self._is_running = True
            self._stage = "ANALYZING"
            self._progress = 10
            self._error_message = None

        def _execute():
            try:
                adapter = model_manager.active_adapter
                pipeline = AnalysisPipeline(adapter)

                with self._lock:
                    self._stage = "INFERENCE"
                    self._progress = 30

                result = pipeline.run(
                    dataset_source=dataset_source,
                    sample_count=sample_count,
                    reducer_type=reducer_type,
                    k_clusters=k_clusters,
                    custom_dir=custom_dir,
                )

                with self._lock:
                    self._last_result = result
                    self._last_run_time = time.time()
                    self._stage = "COMPLETE"
                    self._progress = 100
                    self._is_running = False

                    # Default selections if available
                    if result["blind_spots"] and not self._active_cluster_id:
                        self._active_cluster_id = result["blind_spots"][0]["id"]
                    if result["samples"] and not self._active_sample_id:
                        self._active_sample_id = result["samples"][0]["id"]

                # Record actual analysis completion event
                self.record_event(
                    code="02",
                    title="Evaluation Complete: Manifold Mapped",
                    short="Evaluated",
                    event_type="MANIFOLD_MAPPED",
                    desc=f"Evaluated {result['total_samples']} samples on {result['dataset_name']}. Identified {len(result['blind_spots'])} error clusters.",
                    telemetry=f"Accuracy: {result['metrics']['accuracy']}% | Latency: {result['latency_ms']}ms",
                    accent="amber" if len(result['blind_spots']) > 0 else "green",
                )

                logger.info(
                    f"Analysis complete: {result['total_samples']} samples, "
                    f"{len(result['blind_spots'])} blind spots, accuracy: {result['metrics']['accuracy']}%"
                )
            except Exception as e:
                logger.error(f"Analysis run failed: {e}", exc_info=True)
                with self._lock:
                    self._error_message = str(e)
                    self._stage = "ERROR"
                    self._progress = 0
                    self._is_running = False

        if async_run:
            thread = threading.Thread(target=_execute, daemon=True)
            thread.start()
            return {
                "success": True,
                "message": "Analysis initiated in background.",
                "status": self.get_status(),
            }
        else:
            _execute()
            if self._error_message:
                raise RuntimeError(self._error_message)
            return {
                "success": True,
                "message": "Analysis completed successfully.",
                "summary": self.get_summary(),
            }

    def ensure_initial_analysis(self, sample_count: int = 60) -> None:
        from .model_loader import model_manager
        if not model_manager.is_loaded:
            return
        if self._last_result is None and not self._is_running:
            try:
                self.start_analysis(
                    dataset_source="benchmark",
                    sample_count=sample_count,
                    reducer_type="pca",
                    async_run=False,
                )
            except Exception as e:
                logger.warning(f"Could not complete auto-warmup analysis: {e}")

    def get_summary(self) -> Dict[str, Any]:
        if not self._last_result:
            return {
                "has_data": False,
                "message": "No model loaded. Please load a model to begin analysis.",
                "metrics": None,
                "blind_spots_count": 0,
            }

        return {
            "has_data": True,
            "dataset_name": self._last_result["dataset_name"],
            "total_samples": self._last_result["total_samples"],
            "reducer": self._last_result["reducer"],
            "metrics": self._last_result["metrics"],
            "blind_spots_count": len(self._last_result["blind_spots"]),
            "activations": self._last_result.get("activations", {}),
            "latency_ms": self._last_result["latency_ms"],
            "timestamp": self._last_result["timestamp"],
        }

    def get_blind_spots(self) -> List[Dict[str, Any]]:
        if not self._last_result:
            return []
        return self._last_result["blind_spots"]

    def get_blind_spot_by_id(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        if not self._last_result:
            return None

        spot = next((b for b in self._last_result["blind_spots"] if b["id"] == cluster_id), None)
        if not spot:
            return None

        cluster_samples = [s for s in self._last_result["samples"] if s.get("clusterId") == cluster_id]

        error_confusion = {}
        for s in cluster_samples:
            if not s["correct"]:
                key = f"{s['trueClassName']} -> {s['predictedClassName']}"
                error_confusion[key] = error_confusion.get(key, 0) + 1

        rep_ids = set(spot.get("representative_sample_ids", []))
        representative_samples = [s for s in cluster_samples if s["id"] in rep_ids]
        if not representative_samples:
            representative_samples = cluster_samples[:6]

        return {
            **spot,
            "samples": cluster_samples,
            "representative_samples": representative_samples,
            "error_distribution": [
                {"confusion": k, "count": v}
                for k, v in sorted(error_confusion.items(), key=lambda item: item[1], reverse=True)
            ],
        }

    def get_samples(
        self,
        cluster_id: Optional[str] = None,
        status: Optional[str] = None,
        min_confidence: Optional[float] = None,
        class_id: Optional[int] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if not self._last_result:
            return {"total": 0, "samples": []}

        samples = self._last_result["samples"]

        if cluster_id:
            samples = [s for s in samples if s.get("clusterId") == cluster_id]

        if status and status != "all":
            samples = [s for s in samples if s.get("stability") == status]

        if min_confidence is not None and min_confidence > 0:
            samples = [s for s in samples if s.get("confidence", 0.0) >= min_confidence]

        if class_id is not None:
            samples = [s for s in samples if s.get("trueClass") == class_id or s.get("predictedClass") == class_id]

        total = len(samples)
        paginated = samples[offset : offset + limit]

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "samples": paginated,
        }

    def get_activation_pathway(
        self,
        sample_a_id: Optional[str] = None,
        sample_b_id: Optional[str] = None,
        selected_layer_id: str = "layer-middle",
    ) -> Dict[str, Any]:
        from .model_loader import model_manager
        if not model_manager.is_loaded:
            raise RuntimeError("No model is loaded.")

        if not self._last_result:
            self.ensure_initial_analysis()
        if not self._last_result:
            raise RuntimeError("Analysis engine is not initialized.")

        samples = self._last_result["samples"]
        images = self._last_result.get("sample_images", {})

        successful_pool = [s for s in samples if s.get("correct")]
        failed_pool = [s for s in samples if not s.get("correct")]

        # Select Sample A (Successful / Canonical)
        meta_a = None
        if sample_a_id:
            meta_a = next((s for s in samples if s["id"] == sample_a_id), None)
        if not meta_a:
            if successful_pool:
                meta_a = max(successful_pool, key=lambda s: s.get("confidence", 0.0))
            else:
                meta_a = samples[0]

        # Select Sample B (Failed / Degraded)
        meta_b = None
        if sample_b_id:
            meta_b = next((s for s in samples if s["id"] == sample_b_id), None)
        if not meta_b:
            if failed_pool:
                meta_b = max(failed_pool, key=lambda s: s.get("confidence", 0.0))
            else:
                meta_b = samples[-1]

        img_a = images.get(meta_a["id"])
        img_b = images.get(meta_b["id"])

        if img_a is None or img_b is None:
            gen = BenchmarkDatasetGenerator(seed=42)
            fresh_samples = {s.id: s.image for s in gen.generate(count_per_class=10)}
            img_a = img_a or fresh_samples.get(meta_a["id"])
            img_b = img_b or fresh_samples.get(meta_b["id"])

        adapter = model_manager.active_adapter
        engine = ActivationPathwayEngine(adapter)

        res = engine.compare_pathways(
            img_a=img_a,
            img_b=img_b,
            sample_meta_a=meta_a,
            sample_meta_b=meta_b,
            selected_layer_id=selected_layer_id,
        )

        res["available_successful_samples"] = [
            {
                "id": s["id"],
                "trueClass": s["trueClass"],
                "trueClassName": s["trueClassName"],
                "predictedClass": s["predictedClass"],
                "predictedClassName": s["predictedClassName"],
                "confidence": s["confidence"],
                "correct": s["correct"],
                "stability": s["stability"],
                "perturbation": s.get("perturbation", "clean"),
            }
            for s in successful_pool[:12]
        ]

        res["available_failed_samples"] = [
            {
                "id": s["id"],
                "trueClass": s["trueClass"],
                "trueClassName": s["trueClassName"],
                "predictedClass": s["predictedClass"],
                "predictedClassName": s["predictedClassName"],
                "confidence": s["confidence"],
                "correct": s["correct"],
                "stability": s["stability"],
                "perturbation": s.get("perturbation", "none"),
            }
            for s in failed_pool[:12]
        ]

        return res

    def get_whatif_samples(self) -> List[Dict[str, Any]]:
        from .model_loader import model_manager
        if not model_manager.is_loaded:
            return []

        if not self._last_result:
            self.ensure_initial_analysis()
        if not self._last_result:
            return []

        samples = self._last_result["samples"]
        seen_classes = set()
        candidates = []
        for s in samples:
            cls = s["trueClass"]
            if cls not in seen_classes:
                seen_classes.add(cls)
                candidates.append({
                    "id": s["id"],
                    "trueClass": s["trueClass"],
                    "trueClassName": s["trueClassName"],
                    "confidence": s["confidence"],
                    "correct": s["correct"],
                    "thumbnailGrid": s.get("thumbnailGrid"),
                })
        return candidates

    def run_whatif_experiment(
        self,
        sample_id: Optional[str] = None,
        rotation: float = 0.0,
        noise: float = 0.0,
        blur: float = 0.0,
        crop: float = 0.0,
        brightness: float = 0.0,
        occlusion: float = 0.0,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        from .model_loader import model_manager
        if not model_manager.is_loaded:
            raise RuntimeError("No model is loaded.")

        if not self._last_result:
            self.ensure_initial_analysis()
        if not self._last_result:
            raise RuntimeError("Analysis engine is not initialized.")

        samples = self._last_result["samples"]
        images = self._last_result.get("sample_images", {})

        target_sample = None
        if sample_id:
            target_sample = next((s for s in samples if s["id"] == sample_id), None)
        if not target_sample:
            target_sample = next((s for s in samples if s.get("correct")), samples[0])

        sid = target_sample["id"]
        img = images.get(sid)
        if img is None:
            if model_manager.active_model_type == "custom":
                try:
                    from pathlib import Path
                    import importlib.util
                    ds_path = str(Path(__file__).resolve().parents[3] / "custom" / "dataset.py")
                    spec = importlib.util.spec_from_file_location("custom_dataset_mod", ds_path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        shape_ds = mod.ShapeDataset(size=50, seed=42)
                        idx = int(sid.replace("SMP-", "")) - 1 if "SMP-" in sid else 0
                        idx = max(0, min(idx, len(shape_ds) - 1))
                        tensor, _ = shape_ds[idx]
                        arr = (tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        img = Image.fromarray(arr)
                except Exception:
                    img = None
            if img is None:
                gen = BenchmarkDatasetGenerator(seed=42)
                fresh_samples = {s.id: s.image for s in gen.generate(count_per_class=10)}
                img = fresh_samples.get(sid) or fresh_samples.get("SMP-001")
            if img is None:
                img = Image.new("RGB", (32, 32), (40, 40, 60))

        adapter = model_manager.active_adapter
        engine = WhatIfExperimentEngine(adapter)

        res = engine.run_experiment(
            image=img,
            sample_id=sid,
            rotation=rotation,
            noise=noise,
            blur=blur,
            crop=crop,
            brightness=brightness,
            occlusion=occlusion,
            include_images=include_images,
        )

        # Record event in replay if perturbation was significant
        if abs(rotation) > 0.1 or noise > 0.05 or blur > 0.05 or occlusion > 0.05:
            action_desc = []
            if abs(rotation) > 0.1:
                action_desc.append(f"rotation: {rotation:+.1f}°")
            if noise > 0.05:
                action_desc.append(f"noise: {noise:.1f}%")
            if blur > 0.05:
                action_desc.append(f"blur: {blur:.1f}")

            flip_text = "PREDICTION FLIPPED!" if res["prediction_flip"] else "Prediction held stable"
            self.record_event(
                code="04",
                title=f"Stress Perturbation on #{sid}",
                short="Stressed",
                event_type="STRESS_TEST",
                desc=f"Applied {', '.join(action_desc)}. Result: {flip_text} ({res['original_prediction']} → {res['perturbed_prediction']}).",
                telemetry=f"Confidence Delta: {res['confidence_change_percentage']:+.1f}% | Robustness Index: {res['robustness_score']}/100",
                accent="magenta" if res["prediction_flip"] else "amber",
            )

        return res


analysis_manager = AnalysisManager()
