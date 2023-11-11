import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

# Check if GPUs are available
assert torch.cuda.device_count() >= 4, "This example requires four GPUs"


class ModelParallelCNN(nn.Module):
    def __init__(self):
        super(ModelParallelCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2).to('cuda:0')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2).to('cuda:1')
        self.layer3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2).to('cuda:3')
        # Corrected the input size to the fully connected layer according to pooling and convolution layers
        self.fc = nn.Linear(64 * 3 * 3, 10).to('cuda:2') 
    
    def forward(self, x):
        x = self.layer1(x.to('cuda:0'))
        x = self.pool(F.relu(x))
        x = self.layer2(x.to('cuda:1'))
        x = self.pool(F.relu(x))
        x = self.layer3(x.to('cuda:3'))
        x = self.pool(F.relu(x))
        x = x.view(x.size(0), -1) # Flatten the output
        x = self.fc(x.to('cuda:2'))
        return x


# Initialize the model
model = ModelParallelCNN()

# MNIST Dataset and DataLoader setup
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss().to('cuda:2') # The loss function needs to be on the same GPU as the last layer
optimizer = torch.optim.Adam(model.parameters())

# Training loop
def train(model, train_loader, criterion, optimizer, num_iterations):
    model.train()
    current_iteration = 0
    for epoch in range(1):  # num_epochs would be defined in your main code
        for batch_idx, (data, target) in enumerate(train_loader):
            # Stop after 20 iterations
            if current_iteration >= num_iterations:
                print("Reached 20 iterations. Stopping training.")
                return
            
            optimizer.zero_grad()
            output = model(data.to('cuda:0'))
            loss = criterion(output, target.to('cuda:2'))
            loss.backward()
            optimizer.step()
            current_iteration += 1
            
            print(f"Iteration {current_iteration}: Loss: {loss.item():.6f}")
            
            # Optionally, break here if you want to ensure only 20 iterations irrespective of epochs
            if current_iteration >= num_iterations:
                print("Reached %d iterations. Stopping training.", num_iterations)
                return

# Start training for 20 iterations
train(model, train_loader, criterion, optimizer, num_iterations=1)

