# Redefining the CNN with Reinforcement Learning layer with corrected input sizes
import torch

import torch.nn as nn

import torch.nn.functional as F

class CustomCNNWithRL(nn.Module):
    def __init__(self, input_size):
        super(CustomCNNWithRL, self).__init__()
        # Assuming square input for simplicity
        self.input_size = input_size

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)

        # Compute size of the flattened layer
        self.flattened_size = self.compute_flattened_size()

        # Reinforcement Learning layer (placeholder)
        self.rl = nn.Linear(256, 256)  #  RL scenario

        # Classification head
        self.fc1_class = nn.Linear(self.flattened_size, 1024)
        self.fc2_class = nn.Linear(1024, 512)
        self.fc3_class = nn.Linear(512, 10)  # 10 classes

        # Captioning head
        self.fc1_caption = nn.Linear(self.flattened_size, 1024)
        self.lstm_caption = nn.LSTM(1024, 512, batch_first=True)
        self.fc2_caption = nn.Linear(512, 1000)  # vocabulary size

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def compute_flattened_size(self):
        # Pass a dummy tensor through the convolutional layers to compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.input_size, self.input_size)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten the features for the fully connected layers
        x_flat = x.view(x.size(0), -1)

        # Classification branch
        x_class = F.relu(self.fc1_class(x_flat))
        x_class = F.relu(self.fc2_class(x_class))
        x_class = self.fc3_class(x_class)
        class_output = F.log_softmax(x_class, dim=1)  # log probabilities for classification

        # Captioning branch
        x_caption = F.relu(self.fc1_caption(x_flat))
        x_caption, _ = self.lstm_caption(x_caption.unsqueeze(1))  # assuming single word at a time
        x_caption = x_caption.squeeze(1)
        caption_output = self.fc2_caption(x_caption)

        # Apply RL layer
        rl_class_output = self.rl(class_output)
        rl_caption_output = self.rl(caption_output)

        # Dropout (for the sake of example, applied to the concatenated outputs)
        combined_output = torch.cat((rl_class_output, rl_caption_output), dim=1)
        combined_output = self.dropout(combined_output)

        return combined_output


# Initialize the model with the correct input size
# I use HybridBranchNet model
model_with_rl = CustomCNNWithRL(input_size=224)  # Assuming an input size of 224x224

# Now we can create the dummy input tensor again
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image

# Forward pass through the model with corrected input tensor sizes
output = model_with_rl(dummy_input)

# Print the output shape to verify that it's working correctly
print(output.shape)
