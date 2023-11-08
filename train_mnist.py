import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).to('cuda:0')  # Send layer1 to GPU 0
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).to('cuda:1')  # Send layer2 to GPU 1

        self.fc = nn.Linear(7*7*32, 10).to('cuda:1')  # Fully connected layer on GPU 1

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out
    
# class ModelParallelResNet50(ResNet):
#     def __init__(self, *args, **kwargs):
#         super(ModelParallelResNet50, self).__init__(
#             Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

#         self.seq1 = nn.Sequential(
#             self.conv1,
#             self.bn1,
#             self.relu,
#             self.maxpool,

#             self.layer1,
#             self.layer2
#         ).to('cuda:0')

#         self.seq2 = nn.Sequential(
#             self.layer3,
#             self.layer4,
#             self.avgpool,
#         ).to('cuda:1')

#         self.fc.to('cuda:1')

#     def forward(self, x):
#         x = self.seq2(self.seq1(x).to('cuda:1'))
#         return self.fc(x.view(x.size(0), -1))

# Define data transformations and load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# Initialize the model and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # inputs, labels = inputs.to("cuda:0"), labels

        optimizer.zero_grad()

        outputs = model(inputs.to("cuda:0"))
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

print("Training finished.")
