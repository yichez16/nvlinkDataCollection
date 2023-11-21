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






class ModelParallelCNN(nn.Module):
    def __init__(self, dev0, dev1,dev2, dev3, dev4, dev5, dev6, dev7):
        super(ModelParallelCNN, self).__init__()
        # Define layers
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3).to(dev0) # input 28, f = 3, stride = 1, pd = 0, output 26, C 32
        self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2).to(dev1)  # input 26, f = 2, stride = 2, pd = 0, output 12 C32
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3).to(dev2) # input 12, f = 3, stride = 2, pd = 0, output 12 C64
        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2).to(dev3) # input 12, f = 2, stride = 2, pd = 0, output 5 C64
        self.layer5 = nn.Conv2d(64, 128, kernel_size=3).to(dev4) # input 5, f = 3, stride = 1, pd = 0, output 3 C128
        self.layer6 = nn.Linear(128 * 3 * 3, 128).to(dev5) # input 3*3*128,  output 128
        self.layer7 = nn.Linear(128, 64).to(dev6) # 128,  output 64
        self.layer8 = nn.Linear(64, 10).to(dev7)# input 64,  output 10
        self.relu = nn.ReLU()  # ReLU activation

    def forward(self, x):
        x = self.relu(self.layer1(x.to(dev0)))
        x = self.layer2(x.to(dev1))
        x = self.relu(self.layer3(x.to(dev2)))
        x = self.layer4(x.to(dev3))
        x = self.relu(self.layer5(x.to(dev4)))
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.layer6(x.to(dev5)))
        x = self.relu(self.layer7(x.to(dev6)))
        x = self.layer8(x.to(dev7))
        return x


# setup(4, 2)
# dist.init_process_group("gloo", rank=4, world_size=2)
dev0, dev1, dev2, dev3, dev4, dev5, dev6, dev7 = 0,1,3,2,7,6,4,5
batch_value = int(sys.argv[1])

model = ModelParallelCNN(dev0, dev1, dev2, dev3, dev4, dev5, dev6, dev7)

# MNIST Dataset and DataLoader setup
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_value, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss() # The loss function needs to be on the same GPU as the last layer
optimizer = torch.optim.Adam(model.parameters())

# Training loop
def train(model, train_loader, criterion, optimizer, num_iterations):
    model.train()
    current_iteration = 0
    for epoch in range(10):  # num_epochs would be defined in your main code
        for batch_idx, (data, target) in enumerate(train_loader):
            # Stop after 20 iterations
            if current_iteration >= num_iterations:
                print("Stopping training.")
                return
            data = data.view(data.size(0), -1) # Flatten the images
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
train(model, train_loader, criterion, optimizer, num_iterations=1)

