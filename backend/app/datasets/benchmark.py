import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

from ..services.labels import get_class_label


@dataclass
class BenchmarkSample:
    id: str
    image: Image.Image
    true_class_idx: int
    true_class_name: str
    perturbation: str
    features: Dict[str, float]
    thumbnail_grid: List[List[float]]


class BenchmarkDatasetGenerator:
    """
    Generates a rich, deterministic evaluation dataset of realistic vision samples
    with ImageNet ground-truth targets and authentic visual stress perturbations
    (rotation, gaussian blur, contrast loss, sensor noise, occlusion).
    """

    BENCHMARK_CLASSES = [
        {"idx": 207, "name": "golden retriever", "archetype": "canine_golden"},
        {"idx": 208, "name": "Labrador retriever", "archetype": "canine_labrador"},
        {"idx": 281, "name": "tabby cat", "archetype": "feline_tabby"},
        {"idx": 282, "name": "tiger cat", "archetype": "feline_tiger"},
        {"idx": 817, "name": "sports car", "archetype": "vehicle_sports"},
        {"idx": 511, "name": "convertible", "archetype": "vehicle_convertible"},
        {"idx": 404, "name": "airliner", "archetype": "aerospace_airliner"},
        {"idx": 504, "name": "coffee mug", "archetype": "object_mug"},
        {"idx": 508, "name": "computer keyboard", "archetype": "object_keyboard"},
        {"idx": 920, "name": "traffic light", "archetype": "traffic_light"},
    ]

    PERTURBATION_MODES = [
        "clean",
        "clean",
        "blur_mild",
        "blur_heavy",
        "rotation_light",
        "rotation_heavy",
        "contrast_low",
        "noise_sensor",
        "occlusion_partial",
        "lighting_underexposed",
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _draw_archetype(self, archetype: str, seed: int) -> Image.Image:
        rng = random.Random(seed)
        img = Image.new("RGB", (224, 224), (235, 235, 240))
        draw = ImageDraw.Draw(img)

        if archetype == "canine_golden":
            # Outdoor grass gradient
            for y in range(120, 224):
                green = int(90 + (y - 120) * 0.4)
                draw.line([(0, y), (224, y)], fill=(60, green, 40))
            # Golden canine head & ears
            draw.ellipse([60, 60, 164, 164], fill=(215, 170, 85), outline=(170, 125, 50), width=2)
            draw.polygon([(45, 70), (70, 60), (55, 130)], fill=(190, 145, 65))  # Left ear
            draw.polygon([(179, 70), (154, 60), (169, 130)], fill=(190, 145, 65))  # Right ear
            draw.ellipse([90, 120, 134, 155], fill=(70, 50, 40))  # Muzzle
            draw.ellipse([105, 125, 119, 137], fill=(20, 15, 15))  # Nose
            draw.ellipse([78, 92, 92, 106], fill=(40, 25, 15))  # Left eye
            draw.ellipse([132, 92, 146, 106], fill=(40, 25, 15))  # Right eye

        elif archetype == "canine_labrador":
            # Park ground
            for y in range(130, 224):
                draw.line([(0, y), (224, y)], fill=(80, 120, 70))
            # Dark chocolate / dark lab head
            draw.ellipse([62, 65, 162, 165], fill=(75, 50, 35), outline=(50, 30, 20), width=2)
            draw.polygon([(48, 75), (72, 65), (58, 135)], fill=(60, 40, 25))
            draw.polygon([(176, 75), (152, 65), (166, 135)], fill=(60, 40, 25))
            draw.ellipse([92, 122, 132, 154], fill=(45, 30, 20))
            draw.ellipse([106, 126, 118, 136], fill=(15, 10, 10))
            draw.ellipse([80, 94, 92, 106], fill=(25, 15, 10))
            draw.ellipse([132, 94, 144, 106], fill=(25, 15, 10))

        elif archetype == "feline_tabby":
            # Indoor cozy background
            draw.rectangle([0, 0, 224, 224], fill=(220, 210, 200))
            # Tabby cat head
            draw.ellipse([65, 75, 159, 165], fill=(175, 150, 130), outline=(130, 110, 90), width=2)
            # Pointed ears
            draw.polygon([(65, 80), (80, 35), (100, 70)], fill=(160, 135, 115), outline=(110, 90, 70))
            draw.polygon([(159, 80), (144, 35), (124, 70)], fill=(160, 135, 115), outline=(110, 90, 70))
            # Tabby forehead stripes
            for x_offset in [-16, 0, 16]:
                draw.line([(112 + x_offset, 85), (112 + int(x_offset * 1.3), 110)], fill=(90, 70, 50), width=3)
            # Feline almond eyes
            draw.polygon([(82, 110), (94, 104), (100, 112), (92, 116)], fill=(130, 180, 70))
            draw.polygon([(142, 110), (130, 104), (124, 112), (132, 116)], fill=(130, 180, 70))
            draw.ellipse([90, 108, 94, 112], fill=(10, 10, 10))
            draw.ellipse([130, 108, 134, 112], fill=(10, 10, 10))
            # Whiskers
            draw.line([(60, 136), (20, 130)], fill=(180, 180, 180), width=1)
            draw.line([(60, 142), (22, 146)], fill=(180, 180, 180), width=1)
            draw.line([(164, 136), (204, 130)], fill=(180, 180, 180), width=1)
            draw.line([(164, 142), (202, 146)], fill=(180, 180, 180), width=1)

        elif archetype == "feline_tiger":
            # Amber feline with heavy dark stripes
            draw.rectangle([0, 0, 224, 224], fill=(210, 195, 175))
            draw.ellipse([64, 74, 160, 166], fill=(210, 140, 60), outline=(150, 90, 30), width=2)
            draw.polygon([(64, 80), (82, 32), (102, 70)], fill=(190, 120, 50))
            draw.polygon([(160, 80), (142, 32), (122, 70)], fill=(190, 120, 50))
            # Prominent tiger stripes
            for y_s in [88, 98, 108]:
                draw.arc([75, y_s, 149, y_s + 15], start=190, end=350, fill=(40, 25, 15), width=4)
            draw.ellipse([84, 110, 96, 118], fill=(180, 190, 30))
            draw.ellipse([128, 110, 140, 118], fill=(180, 190, 30))
            draw.ellipse([89, 112, 91, 116], fill=(10, 10, 10))
            draw.ellipse([133, 112, 135, 116], fill=(10, 10, 10))

        elif archetype == "vehicle_sports":
            # Asphalt track
            draw.rectangle([0, 140, 224, 224], fill=(60, 62, 65))
            # Sleek aerodynamic sports chassis (Red)
            chassis_coords = [(30, 145), (55, 115), (105, 95), (160, 98), (195, 130), (200, 152), (25, 152)]
            draw.polygon(chassis_coords, fill=(215, 30, 30), outline=(150, 15, 15), width=2)
            # Windshield
            draw.polygon([(95, 100), (145, 102), (155, 120), (85, 120)], fill=(80, 140, 200))
            # Wheels
            draw.ellipse([45, 135, 80, 170], fill=(25, 25, 25), outline=(180, 180, 180), width=3)
            draw.ellipse([145, 135, 180, 170], fill=(25, 25, 25), outline=(180, 180, 180), width=3)

        elif archetype == "vehicle_convertible":
            # Road
            draw.rectangle([0, 140, 224, 224], fill=(70, 70, 75))
            # Open top convertible body (Blue/Teal)
            body_coords = [(28, 148), (60, 130), (95, 115), (155, 128), (198, 135), (202, 152), (25, 152)]
            draw.polygon(body_coords, fill=(35, 120, 200), outline=(20, 80, 150), width=2)
            # Steep windshield only (no roof)
            draw.polygon([(85, 114), (98, 92), (108, 92), (102, 114)], fill=(120, 180, 230))
            # Headrests visible in open cabin
            draw.ellipse([115, 105, 132, 125], fill=(40, 40, 40))
            # Wheels
            draw.ellipse([45, 136, 78, 169], fill=(20, 20, 20), outline=(190, 190, 190), width=3)
            draw.ellipse([146, 136, 179, 169], fill=(20, 20, 20), outline=(190, 190, 190), width=3)

        elif archetype == "aerospace_airliner":
            # Sky gradient
            for y in range(0, 224):
                b = int(220 + (224 - y) * 0.15)
                draw.line([(0, y), (224, y)], fill=(160, 200, min(255, b)))
            # White aircraft fuselage
            draw.ellipse([20, 95, 185, 135], fill=(245, 245, 250), outline=(180, 185, 195), width=2)
            # Swept wing
            draw.polygon([(85, 115), (45, 180), (75, 180), (125, 115)], fill=(210, 215, 225), outline=(160, 165, 175))
            # Tail fin
            draw.polygon([(25, 100), (15, 45), (40, 45), (55, 100)], fill=(30, 80, 160))
            # Windows row
            for wx in range(65, 155, 12):
                draw.ellipse([wx, 110, wx + 6, 118], fill=(60, 90, 130))

        elif archetype == "object_mug":
            # Kitchen counter surface
            draw.rectangle([0, 140, 224, 224], fill=(200, 180, 160))
            # Ceramic coffee mug body
            draw.rounded_rectangle([68, 65, 156, 175], radius=12, fill=(230, 228, 225), outline=(90, 90, 95), width=3)
            # Side handle loop
            draw.arc([136, 85, 188, 155], start=270, end=90, fill=(90, 90, 95), width=8)
            # Top oval with dark coffee
            draw.ellipse([72, 68, 152, 96], fill=(55, 30, 15), outline=(90, 90, 95), width=2)

        elif archetype == "object_keyboard":
            # Modern desk
            draw.rectangle([0, 0, 224, 224], fill=(225, 225, 230))
            # Keyboard chassis
            draw.rounded_rectangle([25, 55, 199, 165], radius=8, fill=(45, 48, 55), outline=(25, 27, 30), width=3)
            # Key matrix
            for row in range(5):
                ky = 66 + row * 18
                for col in range(9):
                    kx = 34 + col * 17
                    draw.rectangle([kx, ky, kx + 13, ky + 13], fill=(85, 90, 100), outline=(30, 32, 38))

        elif archetype == "traffic_light":
            # City backdrop
            draw.rectangle([0, 0, 224, 224], fill=(195, 205, 215))
            # Black traffic signal casing
            draw.rounded_rectangle([80, 25, 144, 195], radius=10, fill=(30, 32, 35), outline=(15, 15, 18), width=3)
            # Three circular signals: Red, Yellow, Green
            draw.ellipse([92, 38, 132, 78], fill=(235, 40, 40), outline=(140, 20, 20), width=2)
            draw.ellipse([92, 90, 132, 130], fill=(245, 185, 30), outline=(150, 110, 15), width=2)
            draw.ellipse([92, 142, 132, 182], fill=(40, 210, 60), outline=(20, 120, 35), width=2)

        return img

    def _apply_perturbation(
        self, img: Image.Image, p_mode: str, seed: int
    ) -> Tuple[Image.Image, Dict[str, float]]:
        rng = random.Random(seed)
        features = {"rotation": 0.0, "brightness": 1.0, "blur": 0.0}
        out_img = img.copy()

        if p_mode == "clean":
            pass

        elif p_mode == "blur_mild":
            radius = rng.uniform(1.8, 2.8)
            out_img = out_img.filter(ImageFilter.GaussianBlur(radius=radius))
            features["blur"] = round(radius, 2)

        elif p_mode == "blur_heavy":
            radius = rng.uniform(4.5, 6.5)
            out_img = out_img.filter(ImageFilter.GaussianBlur(radius=radius))
            features["blur"] = round(radius, 2)

        elif p_mode == "rotation_light":
            angle = rng.choice([-18.0, -12.0, 12.0, 18.0])
            out_img = out_img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(220, 220, 220))
            features["rotation"] = angle

        elif p_mode == "rotation_heavy":
            angle = rng.choice([-45.0, -35.0, 35.0, 45.0, 85.0])
            out_img = out_img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(220, 220, 220))
            features["rotation"] = angle

        elif p_mode == "contrast_low":
            factor = rng.uniform(0.30, 0.45)
            enhancer = ImageEnhance.Contrast(out_img)
            out_img = enhancer.enhance(factor)
            features["brightness"] = round(factor, 2)

        elif p_mode == "noise_sensor":
            arr = np.array(out_img, dtype=np.float32)
            noise = np.random.RandomState(seed).normal(0, 35.0, arr.shape)
            noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            out_img = Image.fromarray(noisy_arr)
            features["blur"] = 0.5

        elif p_mode == "occlusion_partial":
            draw = ImageDraw.Draw(out_img)
            # Occlusion patch over center/side
            ox = rng.randint(50, 110)
            oy = rng.randint(50, 110)
            draw.rectangle([ox, oy, ox + 65, oy + 65], fill=(30, 30, 30))
            features["blur"] = 0.8

        elif p_mode == "lighting_underexposed":
            factor = rng.uniform(0.35, 0.50)
            enhancer = ImageEnhance.Brightness(out_img)
            out_img = enhancer.enhance(factor)
            features["brightness"] = round(factor, 2)

        return out_img, features

    def _compute_thumbnail_grid(self, img: Image.Image, size: int = 8) -> List[List[float]]:
        """Downsample image to 8x8 luminance values in range [0.0, 1.0]."""
        small = img.convert("L").resize((size, size), Image.Resampling.BOX)
        arr = np.array(small, dtype=np.float32) / 255.0
        return [[round(float(v), 2) for v in row] for row in arr]

    def generate(self, count_per_class: int = 10) -> List[BenchmarkSample]:
        """
        Generates benchmark suite.
        Returns a balanced list of BenchmarkSample instances.
        """
        samples: List[BenchmarkSample] = []
        sample_counter = 1

        for cls_info in self.BENCHMARK_CLASSES:
            cls_idx = cls_info["idx"]
            cls_name = cls_info["name"]
            archetype = cls_info["archetype"]

            for i in range(count_per_class):
                seed = self.seed + sample_counter * 101
                base_img = self._draw_archetype(archetype, seed)

                # Select perturbation mode
                p_mode = self.PERTURBATION_MODES[i % len(self.PERTURBATION_MODES)]
                perturbed_img, features = self._apply_perturbation(base_img, p_mode, seed + 7)

                thumb_grid = self._compute_thumbnail_grid(perturbed_img, size=8)

                sample_id = f"SMP-{sample_counter:03d}"
                samples.append(
                    BenchmarkSample(
                        id=sample_id,
                        image=perturbed_img,
                        true_class_idx=cls_idx,
                        true_class_name=cls_name,
                        perturbation=p_mode,
                        features=features,
                        thumbnail_grid=thumb_grid,
                    )
                )
                sample_counter += 1

        return samples
