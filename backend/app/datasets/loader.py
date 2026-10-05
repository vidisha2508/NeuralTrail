import os
import zipfile
from pathlib import Path
from typing import List, Optional
from PIL import Image
import numpy as np

from .benchmark import BenchmarkSample
from ..services.labels import get_class_label


class DatasetLoader:
    """
    Loads custom user datasets from folders or zip archives.
    Standard layout:
      dataset/
        class_name_or_idx/
          img1.jpg
          img2.png
    """

    @staticmethod
    def load_from_directory(dir_path: str, max_samples: int = 200) -> List[BenchmarkSample]:
        root = Path(dir_path)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")

        samples: List[BenchmarkSample] = []
        counter = 1

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Check for subdirectories (classes)
        subdirs = [d for d in root.iterdir() if d.is_dir()]

        if subdirs:
            for sdir in sorted(subdirs):
                class_label = sdir.name
                try:
                    class_idx = int(class_label)
                except ValueError:
                    class_idx = hash(class_label) % 1000

                for f in sorted(sdir.iterdir()):
                    if f.suffix.lower() in valid_exts:
                        try:
                            img = Image.open(f).convert("RGB").resize((224, 224))
                            thumb = [[round(float(v), 2) for v in row] for row in np.array(img.convert("L").resize((8, 8))) / 255.0]
                            samples.append(
                                BenchmarkSample(
                                    id=f"DIR-{counter:03d}",
                                    image=img,
                                    true_class_idx=class_idx,
                                    true_class_name=get_class_label(class_idx) or class_label,
                                    perturbation="custom",
                                    features={"rotation": 0.0, "brightness": 1.0, "blur": 0.0},
                                    thumbnail_grid=thumb,
                                )
                            )
                            counter += 1
                            if len(samples) >= max_samples:
                                break
                        except Exception:
                            continue
                if len(samples) >= max_samples:
                    break
        else:
            # Flat directory
            for f in sorted(root.iterdir()):
                if f.suffix.lower() in valid_exts:
                    try:
                        img = Image.open(f).convert("RGB").resize((224, 224))
                        thumb = [[round(float(v), 2) for v in row] for row in np.array(img.convert("L").resize((8, 8))) / 255.0]
                        samples.append(
                            BenchmarkSample(
                                id=f"DIR-{counter:03d}",
                                image=img,
                                true_class_idx=0,
                                true_class_name="custom_sample",
                                perturbation="custom",
                                features={"rotation": 0.0, "brightness": 1.0, "blur": 0.0},
                                thumbnail_grid=thumb,
                            )
                        )
                        counter += 1
                        if len(samples) >= max_samples:
                            break
                    except Exception:
                        continue

        return samples
