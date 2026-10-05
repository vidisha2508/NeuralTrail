import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from model import NeuralTrailCNN
from dataset import ShapeDataset


CLASS_NAMES = [
    "circle",
    "square",
    "triangle",
    "cross"
]


def tensor_to_pil(tensor):
    image = tensor.permute(1, 2, 0)
    image = (image * 255).clamp(0, 255).byte()

    return Image.fromarray(image.numpy())


def pil_to_tensor(image):
    # Convert PIL image directly to tensor
    pixels = torch.tensor(
        list(image.getdata()),
        dtype=torch.float32
    )

    pixels = pixels.reshape(
        image.height,
        image.width,
        3
    )

    pixels = pixels.permute(2, 0, 1)

    return pixels / 255.0


def rotate(image, angle=45):
    return image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR
    )


def blur(image):
    return image.filter(
        ImageFilter.GaussianBlur(radius=1.5)
    )


def add_noise(image):
    tensor = pil_to_tensor(image)

    noise = torch.randn_like(tensor) * 0.20

    tensor = torch.clamp(
        tensor + noise,
        0,
        1
    )

    return tensor


def occlude(image):
    image = image.copy()

    pixels = image.load()

    # Hide a central 12x12 region
    for y in range(10, 22):
        for x in range(10, 22):
            pixels[x, y] = (0, 0, 0)

    return image


def predict(model, image):

    # Noise returns a Tensor.
    # Other transformations return PIL images.
    if isinstance(image, torch.Tensor):
        input_tensor = image.unsqueeze(0)
    else:
        input_tensor = pil_to_tensor(image).unsqueeze(0)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, prediction = probabilities.max(
            dim=1
        )

    return (
        prediction.item(),
        confidence.item()
    )


def run_experiment(
    model,
    dataset,
    experiment_name,
    transform
):

    results = []

    correct = 0

    for index, (image_tensor, label) in enumerate(dataset):

        # Convert original tensor to PIL
        original_image = tensor_to_pil(
            image_tensor
        )

        # -----------------------------
        # ORIGINAL PREDICTION
        # -----------------------------

        original_prediction, original_confidence = predict(
            model,
            original_image
        )

        # -----------------------------
        # APPLY PERTURBATION
        # -----------------------------

        transformed_image = transform(
            original_image
        )

        # -----------------------------
        # PERTURBED PREDICTION
        # -----------------------------

        transformed_prediction, transformed_confidence = predict(
            model,
            transformed_image
        )

        is_correct = (
            transformed_prediction == label
        )

        if is_correct:
            correct += 1

        # -----------------------------
        # STORE RESULT
        # -----------------------------

        results.append({

            "sample_id": index,

            "true_label": CLASS_NAMES[label],

            "original": {
                "prediction": CLASS_NAMES[
                    original_prediction
                ],

                "confidence": round(
                    original_confidence * 100,
                    2
                ),

                "correct": (
                    original_prediction == label
                )
            },

            "experiment": experiment_name,

            "perturbed": {
                "prediction": CLASS_NAMES[
                    transformed_prediction
                ],

                "confidence": round(
                    transformed_confidence * 100,
                    2
                ),

                "correct": is_correct
            },

            "prediction_flipped": (
                original_prediction
                != transformed_prediction
            ),

            "confidence_change": round(
                (
                    transformed_confidence
                    - original_confidence
                ) * 100,
                2
            )
        })

    accuracy = (
        100 * correct / len(dataset)
    )

    print()
    print("=" * 50)
    print(experiment_name)
    print("=" * 50)

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print(
        f"Correct:  {correct}/{len(dataset)}"
    )

    return results


def main():

    # --------------------------------
    # DEVICE
    # --------------------------------

    device = torch.device("cpu")

    print("Loading Neural Trail CNN...")

    # --------------------------------
    # LOAD MODEL
    # --------------------------------

    model = NeuralTrailCNN(
        num_classes=4
    )

    model.load_state_dict(
        torch.load(
            "neural_trail_cnn.pth",
            map_location=device
        )
    )

    model.eval()

    # --------------------------------
    # FIXED TEST DATASET
    # --------------------------------

    dataset = ShapeDataset(
        size=400,
        seed=123
    )

    print(
        f"Test samples: {len(dataset)}"
    )

    # --------------------------------
    # RUN EXPERIMENTS
    # --------------------------------

    all_results = {}

    # Normal baseline
    all_results["normal"] = run_experiment(
        model,
        dataset,
        "NORMAL",
        lambda image: image
    )

    # Rotation
    all_results["rotation"] = run_experiment(
        model,
        dataset,
        "ROTATION_45",
        lambda image: rotate(
            image,
            45
        )
    )

    # Gaussian noise
    all_results["noise"] = run_experiment(
        model,
        dataset,
        "GAUSSIAN_NOISE",
        lambda image: add_noise(
            image
        )
    )

    # Blur
    all_results["blur"] = run_experiment(
        model,
        dataset,
        "BLUR",
        lambda image: blur(
            image
        )
    )

    # Occlusion
    all_results["occlusion"] = run_experiment(
        model,
        dataset,
        "OCCLUSION",
        lambda image: occlude(
            image
        )
    )

    # --------------------------------
    # SAVE RESULTS
    # --------------------------------

    results_dir = Path("results")

    results_dir.mkdir(
        exist_ok=True
    )

    output_file = (
        results_dir /
        "stress_results.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )

    # --------------------------------
    # DONE
    # --------------------------------

    print()
    print("=" * 50)
    print("RESULTS SAVED")
    print("=" * 50)

    print(output_file)


if __name__ == "__main__":
    main()