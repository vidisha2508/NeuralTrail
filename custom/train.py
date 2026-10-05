import torch
import torch.nn as nn
import torch.optim as optim

from model import NeuralTrailCNN
from dataset import get_dataloaders


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders()

    model = NeuralTrailCNN(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {running_loss / len(train_loader):.4f} "
            f"Train Accuracy: {accuracy:.2f}%"
        )

    torch.save(model.state_dict(), "neural_trail_cnn.pth")

    print("\nModel saved as neural_trail_cnn.pth")


if __name__ == "__main__":
    train()