import torch
from model import NeuralTrailCNN
from dataset import get_dataloaders


CLASS_NAMES = [
    "circle",
    "square",
    "triangle",
    "cross"
]


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloaders()

    model = NeuralTrailCNN(num_classes=4).to(device)
    model.load_state_dict(
        torch.load(
            "neural_trail_cnn.pth",
            map_location=device
        )
    )

    model.eval()

    correct = 0
    total = 0

    confusion_matrix = torch.zeros(4, 4, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(probabilities, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for true_label, predicted_label in zip(labels, predicted):
                confusion_matrix[
                    true_label,
                    predicted_label
                ] += 1

    accuracy = 100 * correct / total

    print("\n=== NEURAL TRAIL MODEL EVALUATION ===")
    print(f"Test samples: {total}")
    print(f"Correct:      {correct}")
    print(f"Accuracy:     {accuracy:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix)

    print("\nClass names:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i}: {name}")


if __name__ == "__main__":
    evaluate()