# PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Defining Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


# Defining Bottleneck Block (assuming it is a more complex structure, possibly similar to ResNet Bottleneck)
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # Assuming there is a skip connection
        out = self.relu(out)

        return out


# Defining the Fuse Layer
class FuseLayer(nn.Module):
    def __init__(self, channel_sizes):
        super(FuseLayer, self).__init__()
        self.fuse_ops = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) if in_channels != out_channels else nn.Identity()
            for in_channels, out_channels in zip(channel_sizes[:-1], channel_sizes[1:])
        ])

    def forward(self, *inputs):
        return torch.cat([op(input) for op, input in zip(self.fuse_ops, inputs)], dim=1)


# Defining the entire network structure with the provided blocks
class HybridBranchNet(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        # Initial Conv layers as per the provided image
        self.initial_layers = nn.Sequential(
            BasicBlock(3, 64, stride=2),
            BasicBlock(64, 128, stride=2)
        )

        # Bottleneck layers
        self.bottleneck = BottleneckBlock(128, 256, stride=2)

        # Basic and Fuse layers
        self.basic_blocks = nn.ModuleList([BasicBlock(256, 256, stride=2) for _ in range(3)])
        self.fuse_layers = nn.ModuleList([FuseLayer([256, 256, 256]) for _ in range(2)])

        # Final layers
        self.final_layers = nn.Sequential(
            FuseLayer([256, 256, 256]),
            nn.Conv2d(768, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)  # Assuming 10 classes for the Dense layer
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.bottleneck(x)

        # Applying basic blocks and fuse layers
        basic_outs = [block(x) for block in self.basic_blocks]
        x = self.fuse_layers[0](*basic_outs)
        x = self.fuse_layers[1](x, basic_outs[1], basic_outs[2])

        x = self.final_layers(x)
        return x

# Create the custom network instance
custom_network = HybridBranchNet()
# Display the network
print(custom_network)
# Test forward pass
dummy_input = torch.randn(1, 3, 224, 224)
output = custom_network(dummy_input)  # Forward pass
print("Output shape:", output.shape)


# Data preparation and augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the ImageNet dataset
imagenet_data = datasets.ImageFolder('Parcham/train', transform=transform)
data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=32, shuffle=True)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_network.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
custom_network.to(device)

# Training loop
num_epochs = 300  # Define the number of epochs you want to run

for epoch in range(num_epochs):
    custom_network.train()  # Set the model to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = custom_network(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Finished Training')