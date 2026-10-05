import json
from pathlib import Path
from typing import List, Dict

# Standard ImageNet-1K mini dictionary or fallback mapping
DEFAULT_CLASSES: Dict[int, str] = {
    0: "tench", 1: "goldfish", 2: "great white shark", 3: "tiger shark", 4: "hammerhead",
    207: "golden retriever", 208: "Labrador retriever", 254: "pug", 263: "Pembroke Welsh corgi",
    281: "tabby cat", 282: "tiger cat", 285: "Egyptian cat",
    404: "airliner", 504: "coffee mug", 508: "computer keyboard", 511: "convertible",
    515: "corkscrew", 530: "digital clock", 543: "dumbbell", 546: "electric guitar",
    574: "golf ball", 609: "jeep", 620: "laptop", 717: "pickup truck", 779: "school bus",
    812: "space shuttle", 817: "sports car", 859: "toaster", 867: "trailer truck", 920: "traffic light"
}

_imagenet_labels: List[str] = []


def get_all_labels() -> List[str]:
    global _imagenet_labels
    if not _imagenet_labels:
        try:
            from torchvision.models import ResNet18_Weights
            _imagenet_labels = list(ResNet18_Weights.DEFAULT.meta.get("categories", []))
        except Exception:
            pass

    if not _imagenet_labels:
        labels_file = Path(__file__).resolve().parent / "imagenet_classes.json"
        if labels_file.exists():
            try:
                with open(labels_file, "r", encoding="utf-8") as f:
                    _imagenet_labels = json.load(f)
            except Exception:
                pass

    if not _imagenet_labels:
        _imagenet_labels = [DEFAULT_CLASSES.get(i, f"class_{i}") for i in range(1000)]
    return _imagenet_labels


IMAGENET_1K_LABELS: List[str] = get_all_labels()


def get_class_label(index: int) -> str:
    labels = get_all_labels()
    if 0 <= index < len(labels):
        return labels[index]
    return DEFAULT_CLASSES.get(index, f"class_{index}")
