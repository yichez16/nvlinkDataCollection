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


class ModelParallelCNN(nn.Module):
    def __init__(self, dev0, dev1,dev2, dev3, dev4, dev5, dev6, dev7):
        super(ModelParallelCNN, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2).to(dev0)  # MNIST is 1 channel (grayscale)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2).to(dev1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1).to(dev2)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1).to(dev3)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1).to(dev4)
        self.fc1 = nn.Linear(256 * 3 * 3, 4096).to(dev5)  # Adjusted for MNIST size
        self.fc2 = nn.Linear(4096, 4096).to(dev6)
        self.fc3 = nn.Linear(4096, 10).to(dev7)  
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.layer1(x.to(dev0)))
        x = self.layer2(x.to(dev1))
        x = self.relu(self.layer3(x.to(dev2)))
        x = self.layer4(x.to(dev3))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.layer5(x.to(dev4)))
        x = self.relu(self.layer6(x.to(dev5)))
        x = self.layer7(x.to(dev6))

        x = F.relu(self.conv1(x.to(dev0)))
        x = F.max_pool2d(x.to(dev0), kernel_size=3, stride=2)
        x = F.relu(self.conv2(x.to(dev1)))
        x = F.max_pool2d(x.to(dev1), kernel_size=3, stride=2)
        x = F.relu(self.conv3(x.to(dev2)))
        x = F.relu(self.conv4(x.to(dev3)))
        x = F.relu(self.conv5(x.to(dev4)))
        x = F.max_pool2d(x.to(dev4), kernel_size=3, stride=2)
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x.to(dev5)))
        x = F.relu(self.fc2(x.to(dev6)))
        x = self.fc3(x.to(dev7))
        return x


# class AlexNetMNIST(nn.Module):
#     def __init__(self):
#         super(AlexNetMNIST, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # MNIST is 1 channel (grayscale)
#         self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
#         self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(256 * 3 * 3, 4096)  # Adjusted for MNIST size
#         self.fc2 = nn.Linear(4096, 4096)
#         self.fc3 = nn.Linear(4096, 10)  # 10 classes for MNIST

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         x = x.view(-1, 256 * 3 * 3)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# setup(4, 2)
# dist.init_process_group("gloo", rank=4, world_size=2)
dev0, dev1, dev2, dev3, dev4, dev5, dev6, dev7 = 0,1,3,2,7,6,4,5
batch_value = int(sys.argv[1])

model = ModelParallelCNN(dev0, dev1, dev2, dev3, dev4, dev5, dev6, dev7)

# MNIST Dataset and DataLoader setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_value, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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
train(model, train_loader, criterion, optimizer, num_iterations=500)
