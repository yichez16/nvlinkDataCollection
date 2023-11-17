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
    def __init__(self, dev0, dev1,dev2, dev3):
        super(ModelParallelCNN, self).__init__()
        # self.layer1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2).to(dev0)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.layer2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2).to(dev1)
        # self.layer3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2).to(dev2)
        # # Corrected the input size to the fully connected layer according to pooling and convolution layers
        # self.fc_1 = nn.Linear(64 * 3 * 3, 10).to(dev3) 
        self.layer1 = nn.Linear(784, 128).to(dev0)
        self.layer2 = nn.Linear(128, 256).to(dev1)
        self.layer3 = nn.Linear(256, 512).to(dev2)
        self.layer4 = nn.Linear(512, 10).to(dev3)
        
    
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
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_value, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss().to(dev3) # The loss function needs to be on the same GPU as the last layer
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
            data = data.to(dev0)
            target = target.to(dev3)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            current_iteration += 1
            
            print(f"Iteration {current_iteration}: Loss: {loss.item():.6f}")
            
            # Optionally, break here if you want to ensure only 20 iterations irrespective of epochs
            if current_iteration >= num_iterations:
                print("Reached %d iterations. Stopping training." % num_iterations)

                return

# Start training for 20 iterations
train(model, train_loader, criterion, optimizer, num_iterations=20)

