import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN model with model parallelism
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(64 * 7 * 7, 10))

        # Move layers to respective GPUs
        self.layer1 = self.layer1.to("cuda:0")
        self.layer2 = self.layer2.to("cuda:1")
        self.layer3 = self.layer3.to("cuda:2")
        self.layer4 = self.layer4.to("cuda:3")

    def forward(self, x):
        x = self.layer1(x).to("cuda:0")
        x = self.layer2(x).to("cuda:1")
        x = self.layer3(x).to("cuda:2")
        x = x.view(x.size(0), -1).to("cuda:3")  # Flatten 
        x = self.layer4(x).to("cuda:3")
        return x

# Define data transformations and load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# Initialize the model and optimizer
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to("cuda:0"), labels.to("cuda:3")

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

print("Training finished.")
