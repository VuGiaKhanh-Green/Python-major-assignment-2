import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_augmented = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_normal = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_normal)

train_dataset, val_dataset = random_split(dataset, [45000, 5000])
val_dataset.dataset.transform = transform_normal

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = test_dataset.classes

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, epochs=10):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * correct / total)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%, Val Acc: {val_accs[-1]:.2f}%")

    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    acc = 100 * correct / total
    return y_true, y_pred, acc

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, title):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(train_losses, label='Train Loss', color='tab:blue')
    ax1.plot(val_losses, label='Val Loss', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(train_accs, label='Train Acc', color='tab:green', linestyle='--')
    ax2.plot(val_accs, label='Val Acc', color='tab:red', linestyle='--')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(loc='lower right')

    plt.title(title)
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
    plt.title(title)
    plt.show()

cnn_model = CNN()
cnn_train_losses, cnn_val_losses, cnn_train_accs, cnn_val_accs = train_model(cnn_model, train_loader, val_loader)
y_true_cnn, y_pred_cnn, test_acc_cnn = evaluate_model(cnn_model, test_loader)
plot_learning_curves(cnn_train_losses, cnn_val_losses, cnn_train_accs, cnn_val_accs, "CNN Learning Curve")
plot_confusion_matrix(y_true_cnn, y_pred_cnn, "CNN Confusion Matrix")
print("CNN Test Accuracy:", test_acc_cnn)

dataset.transform = transform_normal  # remove augmentation for MLP
mlp_train_dataset, mlp_val_dataset = random_split(dataset, [45000, 5000])
mlp_train_loader = DataLoader(mlp_train_dataset, batch_size=64, shuffle=True)
mlp_val_loader = DataLoader(mlp_val_dataset, batch_size=64, shuffle=False)

mlp_model = MLP()
mlp_train_losses, mlp_val_losses, mlp_train_accs, mlp_val_accs = train_model(mlp_model, mlp_train_loader, mlp_val_loader)
y_true_mlp, y_pred_mlp, test_acc_mlp = evaluate_model(mlp_model, test_loader)
plot_learning_curves(mlp_train_losses, mlp_val_losses, mlp_train_accs, mlp_val_accs, "MLP Learning Curve")
plot_confusion_matrix(y_true_mlp, y_pred_mlp, "MLP Confusion Matrix")
print("MLP Test Accuracy:", test_acc_mlp)


def compare_models(cnn_acc, mlp_acc, cnn_losses, mlp_losses, cnn_accs, mlp_accs):
    epochs = len(cnn_losses)
    x = np.arange(1, epochs + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(x, cnn_losses, label='CNN', color='blue')
    axs[0].plot(x, mlp_losses, label='MLP', color='orange')
    axs[0].set_title('Validation Loss Comparison')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(x, cnn_accs, label='CNN', color='green')
    axs[1].plot(x, mlp_accs, label='MLP', color='red')
    axs[1].set_title('Validation Accuracy Comparison')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle(f'CNN vs MLP\nFinal Test Accuracy: CNN = {cnn_acc:.2f}%, MLP = {mlp_acc:.2f}%', fontsize=14)
    plt.tight_layout()
    plt.show()

compare_models(
    cnn_acc=test_acc_cnn,
    mlp_acc=test_acc_mlp,
    cnn_losses=cnn_val_losses,
    mlp_losses=mlp_val_losses,
    cnn_accs=cnn_val_accs,
    mlp_accs=mlp_val_accs
)

import pandas as pd

comparison_df = pd.DataFrame({
    'Model': ['CNN', 'MLP'],
    'Final Train Accuracy (%)': [cnn_train_accs[-1], mlp_train_accs[-1]],
    'Final Val Accuracy (%)': [cnn_val_accs[-1], mlp_val_accs[-1]],
    'Test Accuracy (%)': [test_acc_cnn, test_acc_mlp],
    'Final Train Loss': [cnn_train_losses[-1], mlp_train_losses[-1]],
    'Final Val Loss': [cnn_val_losses[-1], mlp_val_losses[-1]],
    'Epochs': [len(cnn_train_losses), len(mlp_train_losses)]
})

print(comparison_df)

