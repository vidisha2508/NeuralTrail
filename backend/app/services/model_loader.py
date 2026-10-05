import os
from pathlib import Path
from typing import Dict, Type, Optional, List, Any
from ..adapters.base import BaseModelAdapter
from ..adapters.registry import ModelRegistry
from ..adapters.generic_pytorch import GenericPyTorchAdapter
from ..config import settings


class ModelManager:
    """
    Central Model Manager for Neural Trail.
    Manages active model lifecycle, switching, checkpoint resolution, and state resets.
    Strictly follows:
      - Starts in an unloaded state (honest empty state).
      - Switching models resets the investigation state cleanly.
      - Provides ResNet18, MobileNet, and GenericPyTorchAdapter.
    """

    def __init__(self):
        self._active_adapter: Optional[BaseModelAdapter] = None
        self._active_model_type: Optional[str] = None
        self._active_checkpoint_path: Optional[str] = None

    @property
    def active_adapter(self) -> Optional[BaseModelAdapter]:
        return self._active_adapter

    @property
    def active_model_type(self) -> Optional[str]:
        return self._active_model_type

    @property
    def is_loaded(self) -> bool:
        return self._active_adapter is not None and self._active_adapter.metadata().get("is_loaded", False)

    def unload_model(self) -> None:
        """Unloads current model and resets investigation state."""
        self._active_adapter = None
        self._active_model_type = None
        self._active_checkpoint_path = None
        from .analysis_manager import analysis_manager
        analysis_manager.reset()

    def load_model(
        self,
        model_type: str = "resnet18",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> BaseModelAdapter:
        model_key = model_type.lower()
        adapter_cls = ModelRegistry.get_adapter_class(model_key)
        device_to_use = device or settings.DEVICE

        models_dir = Path(__file__).resolve().parents[2] / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        if model_key == "resnet18":
            path_to_use = checkpoint_path or str(models_dir / "resnet18.pth")
            adapter = adapter_cls(checkpoint_path=path_to_use, device=device_to_use)
        elif model_key == "mobilenet":
            path_to_use = checkpoint_path or str(models_dir / "mobilenet_v2.pth")
            adapter = adapter_cls(checkpoint_path=path_to_use, device=device_to_use)
        else:
            path_to_use = checkpoint_path or str(models_dir / "custom.pth")
            adapter = adapter_cls(checkpoint_path=path_to_use, device=device_to_use)

        self._active_adapter = adapter
        self._active_model_type = model_key
        self._active_checkpoint_path = path_to_use

        # Reset and refresh investigation state for the newly loaded model
        from .analysis_manager import analysis_manager
        analysis_manager.reset()

        meta = adapter.metadata()
        analysis_manager.record_event(
            code="01",
            title=f"Model Loaded: {meta['model_name']}",
            short="Loaded",
            event_type="MODEL_LOADED",
            desc=f"{meta['architecture']} successfully initialized on {meta['device'].upper()} runtime.",
            telemetry=f"Parameters: {meta['total_parameters']:,} | Classes: {meta['num_classes']} | Resolution: {meta['input_resolution']}",
            accent="cyan",
        )

        # Run fresh evaluation
        analysis_manager.start_analysis(sample_count=100, async_run=False)

        return self._active_adapter

    def load_custom_test_model(self) -> BaseModelAdapter:
        """
        Loads the test model from custom/ (custom/model.py + custom/neural_trail_cnn.pth)
        with ShapeDataset (custom/dataset.py) to validate the generic PyTorch pipeline.
        """
        root_dir = Path(__file__).resolve().parents[3]
        custom_dir = root_dir / "custom"
        ckpt_path = str(custom_dir / "neural_trail_cnn.pth")
        model_def_path = str(custom_dir / "model.py")
        dataset_path = str(custom_dir / "dataset.py")

        adapter = GenericPyTorchAdapter(
            checkpoint_path=ckpt_path,
            model_def_path=model_def_path,
            class_name="NeuralTrailCNN",
            input_shape=(3, 32, 32),
            classes=["circle", "square", "triangle", "cross"],
            device=settings.DEVICE,
            custom_name="NeuralTrailCNN (Custom Test Model)",
            dataset_path=dataset_path,
        )

        self._active_adapter = adapter
        self._active_model_type = "custom"
        self._active_checkpoint_path = ckpt_path

        from .analysis_manager import analysis_manager
        analysis_manager.reset()

        meta = adapter.metadata()
        analysis_manager.record_event(
            code="01",
            title=f"Custom Model Loaded: {meta['model_name']}",
            short="Loaded",
            event_type="MODEL_LOADED",
            desc=f"Dynamically loaded {meta['architecture']} from custom/model.py with ShapeDataset evaluation.",
            telemetry=f"Parameters: {meta['total_parameters']:,} | Classes: {meta['num_classes']} ({', '.join(meta.get('classes', []))}) | Input: 3×32×32",
            accent="magenta",
        )

        # Run fresh evaluation on ShapeDataset
        analysis_manager.start_analysis(sample_count=100, async_run=False)

        return self._active_adapter

    def load_uploaded_model(
        self,
        file_bytes: bytes,
        filename: str,
        model_code: Optional[str] = None,
        class_name: Optional[str] = None,
        input_shape: Optional[List[int]] = None,
        classes: Optional[List[str]] = None,
    ) -> BaseModelAdapter:
        uploads_dir = Path(__file__).resolve().parents[2] / "models" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        save_path = uploads_dir / filename
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        model_def_path = None
        if model_code:
            code_path = uploads_dir / f"{save_path.stem}_def.py"
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(model_code)
            model_def_path = str(code_path)

        shape_tuple = tuple(input_shape) if (input_shape and len(input_shape) == 3) else (3, 224, 224)
        model_name = Path(filename).stem.replace("_", " ").title()

        adapter = GenericPyTorchAdapter(
            checkpoint_path=str(save_path),
            model_def_path=model_def_path,
            class_name=class_name,
            input_shape=shape_tuple,
            classes=classes or ["Class_0", "Class_1", "Class_2", "Class_3"],
            device=settings.DEVICE,
            custom_name=f"Custom: {model_name}",
        )

        self._active_adapter = adapter
        self._active_model_type = "custom"
        self._active_checkpoint_path = str(save_path)

        from .analysis_manager import analysis_manager
        analysis_manager.reset()

        meta = adapter.metadata()
        analysis_manager.record_event(
            code="01",
            title=f"Uploaded Model: {meta['model_name']}",
            short="Uploaded",
            event_type="MODEL_LOADED",
            desc=f"Uploaded PyTorch model checkpoint ({filename}) successfully registered.",
            telemetry=f"Parameters: {meta['total_parameters']:,} | Shape: {meta['input_resolution']}",
            accent="magenta",
        )

        analysis_manager.start_analysis(sample_count=100, async_run=False)

        return self._active_adapter

    def list_checkpoints(self) -> List[str]:
        models_dir = Path(__file__).resolve().parents[2] / "models"
        if not models_dir.exists():
            return []
        pts = [str(p) for p in models_dir.glob("*.pth")] + [str(p) for p in models_dir.glob("*.pt")]
        uploads_dir = models_dir / "uploads"
        if uploads_dir.exists():
            pts += [str(p) for p in uploads_dir.glob("*.pth")] + [str(p) for p in uploads_dir.glob("*.pt")]
        return pts


model_manager = ModelManager()
