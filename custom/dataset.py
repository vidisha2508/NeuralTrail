import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw


class ShapeDataset(Dataset):
    def __init__(self, size=2000, image_size=32, seed=42):
        self.size = size
        self.image_size = image_size
        self.seed = seed

        self.classes = ["circle", "square", "triangle", "cross"]

        # Generate the dataset once so it stays fixed
        self.samples = []

        rng = random.Random(seed)

        for index in range(size):
            image = self.generate_shape(index, rng)
            self.samples.append(image)

    def generate_shape(self, index, rng):
        image = Image.new(
            "RGB",
            (self.image_size, self.image_size),
            (0, 0, 0)
        )

        draw = ImageDraw.Draw(image)

        shape = index % 4

        x = rng.randint(7, 17)
        y = rng.randint(7, 17)
        size = rng.randint(6, 12)

        if shape == 0:
            draw.ellipse(
                (x, y, x + size, y + size),
                fill=(255, 255, 255)
            )

        elif shape == 1:
            draw.rectangle(
                (x, y, x + size, y + size),
                fill=(255, 255, 255)
            )

        elif shape == 2:
            draw.polygon(
                [
                    (x + size // 2, y),
                    (x, y + size),
                    (x + size, y + size)
                ],
                fill=(255, 255, 255)
            )

        elif shape == 3:
            thickness = max(2, size // 3)

            draw.rectangle(
                (
                    x + size // 2 - thickness,
                    y,
                    x + size // 2 + thickness,
                    y + size
                ),
                fill=(255, 255, 255)
            )

            draw.rectangle(
                (
                    x,
                    y + size // 2 - thickness,
                    x + size,
                    y + size // 2 + thickness
                ),
                fill=(255, 255, 255)
            )

        # Convert PIL image → PyTorch tensor
        image = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32
        )

        image = image.reshape(
            self.image_size,
            self.image_size,
            3
        )

        image = image.permute(2, 0, 1)

        image = image / 255.0

        return image

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = self.samples[index]
        label = index % 4

        return image, label


def get_dataloaders(batch_size=64):

    train_dataset = ShapeDataset(
        size=2000,
        seed=42
    )

    test_dataset = ShapeDataset(
        size=400,
        seed=123
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader