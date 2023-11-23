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
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class RandomLabelDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Generate a random label
        label = random.randint(0, 999)  # Assuming 1000 classes, labeled 0 to 999

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset
train_dataset = RandomLabelDataset('data', transform=transform)



# class RandomImageNetDataset(Dataset):
#     def __init__(self, num_samples=1000, num_classes=1000, image_size=(224, 224, 3)):
#         self.num_samples = num_samples
#         self.num_classes = num_classes
#         self.image_size = image_size

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         # Generate a random image (noise)
#         image = np.random.rand(*self.image_size).astype(np.float32)
#         # Normalize the image
#         image = (image - 0.485) / 0.229
#         # Permute the dimensions to match PyTorch's expectation: Channels x Height x Width
#         image = np.transpose(image, (2, 0, 1))
#         # Generate a random label
#         label = np.random.randint(0, self.num_classes)
#         return torch.tensor(image), torch.tensor(label)

# # Create the dummy datasets
# train_dataset = RandomImageNetDataset()



class ModelParallelCNN(nn.Module):
    def __init__(self, dev0, dev1,dev2, dev3, dev4, dev5, dev6, dev7):
        super(ModelParallelCNN, self).__init__()
        # Define layers
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1).to(dev0) # input 224, 224 -3+2 +1 = 224
        self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2).to(dev1)  # input 26, f = 2, stride = 2, pd = 0, output (224 -2)/2 + 1 = 112
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3, padding=1).to(dev2) # input 112,  112-3+2+1 = 112
        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2).to(dev3) # input 112, 56
        self.layer5 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(dev4) # input 56, 56
        self.layer6 = nn.MaxPool2d(kernel_size=2, stride=2).to(dev5) # input 56, 28
        self.layer7 = nn.Linear(128 * 28 * 28 , 1024).to(dev6) # input 3*3*128,  output 128
        self.layer8 = nn.Linear(1024, 1000).to(dev7)# input 64,  output 10
        self.relu = nn.ReLU()  # ReLU activation

    def forward(self, x):
        x = self.relu(self.layer1(x.to(dev0)))
        x = self.layer2(x.to(dev1))
        x = self.relu(self.layer3(x.to(dev2)))
        x = self.layer4(x.to(dev3))
        x = self.relu(self.layer5(x.to(dev4)))
        x = self.layer6(x.to(dev5))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.layer7(x.to(dev6)))
        x = self.relu(self.layer8(x.to(dev7)))
        return x


# setup(4, 2)
# dist.init_process_group("gloo", rank=4, world_size=2)
dev0, dev1, dev2, dev3, dev4, dev5, dev6, dev7 = 0,1,3,2,7,6,4,5
batch_value = int(sys.argv[1])

model = ModelParallelCNN(dev0, dev1, dev2, dev3, dev4, dev5, dev6, dev7)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_value, shuffle=True)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
def train(model, train_loader, criterion, optimizer, num_iterations):
    model.train()
    current_iteration = 0
    for epoch in range(100000):  # num_epochs would be defined in your main code
        for batch_idx, (data, target) in enumerate(train_loader):
            # Stop after 20 iterations
            if current_iteration >= num_iterations:
                print("Stopping training.")
                return
            # data = data.view(data.size(0), -1) # Flatten the images required for mlp
            optimizer.zero_grad()
            output = model(data.to(dev0))
            loss = criterion(output, target.to(dev7))
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

