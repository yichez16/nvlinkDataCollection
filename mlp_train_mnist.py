import torch
import os
import sys
import tempfile
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RandomImageNetDataset(Dataset):
    def __init__(self, num_samples=1000000, num_classes=1000, image_size=(224, 224, 3)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image (noise)
        image = np.random.rand(*self.image_size).astype(np.float32)
        
        # Normalize the image
        image = (image - 0.485) / 0.229

        # Generate a random label
        label = np.random.randint(0, self.num_classes)

        return torch.tensor(image), torch.tensor(label)

# Create the dummy datasets
train_dataset = RandomImageNetDataset()


class ModelParallelCNN(nn.Module):
    def __init__(self, dev0, dev1,dev2, dev3):
        super(ModelParallelCNN, self).__init__()
        # self.layer1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2).to(dev0)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.layer2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2).to(dev1)
        # self.layer3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2).to(dev2)
        # # Corrected the input size to the fully connected layer according to pooling and convolution layers
        # self.fc_1 = nn.Linear(64 * 3 * 3, 10).to(dev3) 
        self.layer1 = nn.Linear(224*224*3, 1024).to(dev0)
        self.layer2 = nn.Linear(1024, 2048).to(dev1)
        self.layer3 = nn.Linear(2048, 4096).to(dev2)
        self.layer4 = nn.Linear(4096, 1000).to(dev3)
        
    
    def forward(self, x):
        # Ensure input is on device 0 and pass through the first layer
        x = x.to(dev0)
        x = F.relu(self.layer1(x))

        # Move to device 1 for the second layer
        x = x.to(dev1)
        x = F.relu(self.layer2(x))

        # Move to device 2 for the third layer
        x = x.to(dev2)
        x = F.relu(self.layer3(x))

        # Move to device 3 for the fourth layer
        x = x.to(dev3)
        x = F.relu(self.layer4(x))
        return x


# Initialize the model
# setup(4, 2)
# dist.init_process_group("gloo", rank=4, world_size=2)
dev0, dev1, dev2, dev3 = 0,1,3,2 #0, 1, 3, 2
batch_value = int(sys.argv[1])

model = ModelParallelCNN(dev0, dev1, dev2, dev3)

# MNIST Dataset and DataLoader setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_value, shuffle=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_value, shuffle=True)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss().to(dev3) # The loss function needs to be on the same GPU as the last layer
optimizer = torch.optim.Adam(model.parameters())

# Training loop
def train(model, train_loader, criterion, optimizer, num_iterations):
    model.train()
    current_iteration = 0
    for epoch in range(100):  # num_epochs would be defined in your main code
        for batch_idx, (data, target) in enumerate(train_loader):
            # Stop after 20 iterations
            if current_iteration >= num_iterations:
                print("Stopping training.")
                return
            data = data.view(data.size(0), -1) # Flatten the images
            optimizer.zero_grad()
            output = model(data.to(dev0))
            loss = criterion(output, target.to(dev3))
            loss.backward()
            optimizer.step()
            current_iteration += 1
            
            print(f"Iteration {current_iteration}: Loss: {loss.item():.6f}")
            
            # Optionally, break here if you want to ensure only 20 iterations irrespective of epochs
        if current_iteration >= num_iterations:
            print("Reached %d iterations. Stopping training." % num_iterations)

            return

# Start training for 20 iterations
train(model, train_loader, criterion, optimizer, num_iterations=100)

