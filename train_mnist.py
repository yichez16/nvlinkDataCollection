import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define the CNN Model with model parallelism
class ParallelCNN(nn.Module):
    def __init__(self):
        super(ParallelCNN, self).__init__()
        # Define layer 1 and place it on GPU 0
        self.layer1 = nn.Conv2d(1, 20, kernel_size=5).to('cuda:0')
        # Define layer 2 and place it on GPU 1
        self.layer2 = nn.Conv2d(20, 50, kernel_size=5).to('cuda:1')
        # Define layer 3 and place it on GPU 2
        self.fc = nn.Linear(800, 10).to('cuda:2')
        
    def forward(self, x):
        # Move input to GPU 0
        x = x.to('cuda:0')
        x = self.layer1(x)
        # Move tensor to next GPU
        x = x.to('cuda:1')
        x = self.layer2(x)
        x = x.view(-1, 800)
        # Move tensor to next GPU
        x = x.to('cuda:2')
        x = self.fc(x)
        return x

# 2. Load the MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 3. Initialize the model and optimizer
model = ParallelCNN()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 4. Define the training loop
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Assume the target is on the CPU, move it to GPU 2 where the last layer is
        target = target.to('cuda:2')
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
# 5. Run the training process
for epoch in range(1, 11):
    train(model, 'cuda', train_loader, optimizer, epoch)
