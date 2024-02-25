import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchsummary import summary
import os, json
from time import time
from PIL import Image
import numpy as np

print(torch.__version__)

class TinyVGG(nn.Module):
    def __init__(self, filters=10):
        super(TinyVGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, filters, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(filters * 16 * 16, NUM_CLASS)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
    
    
# Define dataset class
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, class_dict, file_dict, transform=None):
        self.root_dir = root_dir
        self.class_dict = class_dict
        self.file_dict = file_dict
        self.transform = transform

    def __len__(self):
        return len(self.file_dict)

    def __getitem__(self, idx):
        file_name = list(self.file_dict.keys())[idx]
        img_path = self.file_dict[file_name]['path']

        # Load image using PIL
        image = Image.open(img_path).convert('RGB')

        # Convert torch.Tensor to PIL Image
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        label = int(self.file_dict[file_name]['class'])

        return image, label

# Load class_dict and file_dict from JSON files
with open('class_dict.json', 'r') as f:
    class_dict = json.load(f)

with open('file_dict.json', 'r') as f:
    file_dict = json.load(f)

# Set random seed for reproducibility
torch.manual_seed(42)

# Constants
WIDTH = 64
HEIGHT = 64
EPOCHS = 100
PATIENCE = 50
LR = 0.001
NUM_CLASS = 10
BATCH_SIZE = 32

# Data transformations
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor()
])

# Create datasets and dataloaders
train_dataset = TinyImageNetDataset(root_dir='./train', class_dict=class_dict, file_dict=file_dict, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Validation and test datasets can be created similarly
val_dataset = TinyImageNetDataset(root_dir='./test', class_dict=class_dict, file_dict=file_dict, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


test_dataset = TinyImageNetDataset(root_dir='./test', class_dict=class_dict, file_dict=file_dict, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Calculate mean and std using torchvision.transforms.Normalize
mean = torch.zeros(3)
std = torch.zeros(3)

for images, _ in train_loader:
    # Images are expected to be in the range [0, 1]
    mean += images.mean(dim=(0, 2, 3))
    std += images.std(dim=(0, 2, 3))

# Calculate the mean and std over the entire dataset
mean /= len(train_loader)
std /= len(train_loader)

print("Mean:", mean.tolist())
print("Std:", std.tolist())

# Instantiate the model
tiny_vgg = TinyVGG()

# Print model summary
summary(tiny_vgg, (3, HEIGHT, WIDTH))

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(tiny_vgg.parameters(), lr=LR)

# Training loop
no_improvement_epochs = 0
best_val_loss = float('inf')
start_time = time()

for epoch in range(EPOCHS):
    # Train
    tiny_vgg.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = tiny_vgg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    tiny_vgg.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = tiny_vgg(images)
            val_loss += criterion(outputs, labels).item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        no_improvement_epochs = 0
        best_val_loss = val_loss
        # Save the best model
        torch.save(tiny_vgg.state_dict(), 'trained_vgg_best1.pth')
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= PATIENCE:
        print(f'Early stopping at epoch = {epoch}')
        break

print(f'\nFinished training, used {(time() - start_time) / 60:.4f} mins.')

# Load the best model
best_model = TinyVGG()
best_model.load_state_dict(torch.load('trained_vgg_best1.pth'))

# Test on hold-out test images
test_loss = 0.0
correct = 0
total = 0

best_model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        outputs = best_model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nTest Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {(correct / total) * 100:.4f}%")