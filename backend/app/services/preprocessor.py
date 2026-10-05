import io
from typing import Union
from PIL import Image
import torch
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image_input: Union[bytes, io.BytesIO, Image.Image]) -> torch.Tensor:
        """
        Converts bytes, file-like object, or PIL Image into normalized PyTorch tensor (1, 3, 224, 224).
        """
        if isinstance(image_input, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image)
        # Add batch dimension: (1, 3, H, W)
        return tensor.unsqueeze(0)

preprocessor = ImagePreprocessor()
