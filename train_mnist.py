import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Check if GPUs are available
assert torch.cuda.device_count() >= 4, "This example requires four GPUs"

# Define the CNN Model with model parallelism
class ModelParallelCNN(nn.Module):
    def __init__(self):
        super(ModelParallelCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2).to('cuda:0')
        self.layer2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2).to('cuda:1')
        self.layer3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2).to('cuda:2')
        self.fc = nn.Linear(7*7*64, 10).to('cuda:3')
    
    def forward(self, x):
        x = x.to('cuda:0')
        x = self.layer1(x)
        x = x.to('cuda:1')
        x = self.layer2(x)
        x = x.to('cuda:2')
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # Flatten the output
        x = x.to('cuda:3')
        x = self.fc(x)
        return x

# Initialize the model
model = ModelParallelCNN()

# MNIST Dataset and DataLoader setup
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss().to('cuda:3') # The loss function needs to be on the same GPU as the last layer
optimizer = torch.optim.Adam(model.parameters())

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.to('cuda:3'))
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

# Start training
train(model, train_loader, criterion, optimizer, num_epochs=5)
