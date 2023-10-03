import torch
import torch.nn as nn
import torch.nn.functional as F


class RNASecondaryStructureCAE(nn.Module):
    def __init__(self, num_hidden_channels, num_channels=8):
        """
        args:
        num_channels: length of the one-hot encoding vector
        num_hidden_channels: number of channels in the hidden layers of both encoder and decoder
        """
        super(RNASecondaryStructureCAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_hidden_channels,
                out_channels=num_hidden_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=num_hidden_channels * 2,
                out_channels=num_hidden_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=num_hidden_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


"""
################################
Example usage
################################
# Define parameters and create instance of the autoencoder
num_hidden_channels: 16 # Can be modified as needed
cae = RNASecondaryStructureCAE(num_hidden_channels)


# Define the binary cross-entropy loss function
criterion = nn.BCEloss()


# Define an optimizer
optimizer = torch.optim.Adam(cae.parameters(), lr=0.001)


########################
Assuming that a dataloader has been created for the data
dataloader = DataLoader(...)

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for data in dataloader: 
    inputs, _ = data    # Assuming data is labeled
    
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = cae(inputs)
    
    # Reshape input and output to match BCE loss (sanity check for sizes)
    inputs = inputs.view(-1, num_input_channels)
    outputs = outputs.view(-1, num_input_channels)
    
    # Compute BCE loss
    loss = criterion(outputs, inputs)
    
    # Backpropagation and optimization
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader).4f}")
    
print("Training finished")
"""
