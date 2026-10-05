import io
import base64
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np


class BasePerturbation(ABC):
    """
    Abstract interface for modular visual perturbations.
    """

    @abstractmethod
    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        """
        Applies transformation to PIL Image given an intensity scalar parameter.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class RotationPerturbation(BasePerturbation):
    """
    Affine 2D in-plane rotation.
    Intensity: angle in degrees (e.g. -45.0 to +45.0).
    """

    @property
    def name(self) -> str:
        return "rotation"

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        if abs(intensity) < 0.5:
            return image
        return image.rotate(
            angle=float(intensity),
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(235, 235, 240),
        )


class GaussianNoisePerturbation(BasePerturbation):
    """
    Additive sensor Gaussian noise.
    Intensity: percentage from 0 to 100.
    """

    @property
    def name(self) -> str:
        return "noise"

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        percent = max(0.0, min(100.0, float(intensity)))
        if percent < 1.0:
            return image

        # Standard deviation sigma scales with percentage (max sigma ~ 50.0)
        sigma = (percent / 100.0) * 50.0
        arr = np.array(image, dtype=np.float32)
        noise = np.random.RandomState(42).normal(0.0, sigma, arr.shape)
        noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_arr)


class BlurPerturbation(BasePerturbation):
    """
    Optical / Motion Gaussian kernel blur.
    Intensity: blur radius in pixels (0.0 to 12.0).
    """

    @property
    def name(self) -> str:
        return "blur"

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        radius = max(0.0, float(intensity))
        if radius < 0.2:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class CropPerturbation(BasePerturbation):
    """
    Scale / Center crop simulating zoom and framing loss.
    Intensity: crop percentage from 0 to 50.
    """

    @property
    def name(self) -> str:
        return "crop"

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        percent = max(0.0, min(50.0, float(intensity)))
        if percent < 1.0:
            return image

        w, h = image.size
        # Margin to crop from each side
        margin_w = int(w * (percent / 200.0))
        margin_h = int(h * (percent / 200.0))

        cropped = image.crop((margin_w, margin_h, w - margin_w, h - margin_h))
        return cropped.resize((w, h), Image.Resampling.BILINEAR)


class BrightnessPerturbation(BasePerturbation):
    """
    Illumination shift (overexposure / underexposure).
    Intensity: percentage from -80 to +80.
    """

    @property
    def name(self) -> str:
        return "brightness"

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        percent = max(-80.0, min(80.0, float(intensity)))
        if abs(percent) < 1.0:
            return image

        factor = 1.0 + (percent / 100.0)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)


class OcclusionPerturbation(BasePerturbation):
    """
    Simulated sensor occlusion or foreign object block.
    Intensity: occlusion percentage of canvas dimension (0 to 60).
    """

    @property
    def name(self) -> str:
        return "occlusion"

    def apply(self, image: Image.Image, intensity: float) -> Image.Image:
        percent = max(0.0, min(60.0, float(intensity)))
        if percent < 2.0:
            return image

        out_img = image.copy()
        draw = ImageDraw.Draw(out_img)
        w, h = out_img.size

        # Patch dimensions
        patch_w = int(w * (percent / 100.0))
        patch_h = int(h * (percent / 100.0))

        # Position occlusion patch over upper-left center quadrant
        x0 = int(w * 0.2)
        y0 = int(h * 0.2)
        x1 = min(w, x0 + patch_w)
        y1 = min(h, y0 + patch_h)

        draw.rectangle([x0, y0, x1, y1], fill=(20, 20, 25), outline=(5, 5, 8), width=1)
        return out_img


class PerturbationComposer:
    """
    Modular registry and composer for perturbation pipelines.
    Allows easy addition of new transformation types.
    """

    def __init__(self):
        self._registry: Dict[str, BasePerturbation] = {
            "rotation": RotationPerturbation(),
            "noise": GaussianNoisePerturbation(),
            "blur": BlurPerturbation(),
            "crop": CropPerturbation(),
            "brightness": BrightnessPerturbation(),
            "occlusion": OcclusionPerturbation(),
        }

    def register(self, perturbation: BasePerturbation) -> None:
        self._registry[perturbation.name.lower()] = perturbation

    def compose(self, image: Image.Image, params: Dict[str, float]) -> Image.Image:
        """
        Applies all specified perturbations in sequential order.
        """
        transformed = image.copy()

        # Deterministic application sequence
        sequence = ["rotation", "crop", "brightness", "blur", "noise", "occlusion"]
        for key in sequence:
            if key in params and key in self._registry:
                intensity = float(params[key])
                transformed = self._registry[key].apply(transformed, intensity)

        # Apply any additional custom registered transformations
        for key, val in params.items():
            if key not in sequence and key in self._registry:
                transformed = self._registry[key].apply(transformed, float(val))

        return transformed


def image_to_base64_data_url(image: Image.Image, format: str = "PNG", quality: int = 85) -> str:
    """Encodes PIL Image to browser-renderable data URL safely across any image mode."""
    buffered = io.BytesIO()
    fmt = format.upper()
    if fmt == "JPEG":
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=quality)
        mime = "jpeg"
    else:
        if image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        image.save(buffered, format="PNG")
        mime = "png"
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{mime};base64,{img_str}"


perturbation_composer = PerturbationComposer()
