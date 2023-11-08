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

        # Initialize layers
        self.layer1.apply(self.init_weights)
        self.layer2.apply(self.init_weights)
        self.layer3.apply(self.init_weights)
        self.layer4.apply(self.init_weights)

    def forward(self, x):
        x = self.layer1(x).to("cuda:0")
        x = x.to("cuda:1")
        x = self.layer2(x).to("cuda:2")
        x = x.to("cuda:3")
        x = self.layer3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.layer4(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

# Define data transformations and load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# Initialize the model and optimizer
model = Model()
model = model.to("cuda:0")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

print("Training finished.")
