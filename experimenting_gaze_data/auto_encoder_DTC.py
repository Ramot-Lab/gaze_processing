import torch
import torch.nn as nn

# Encoder class
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        # Initial convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5), padding=(0, 2))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)  # Leaky ReLU
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))  # MaxPooling

        # BiLSTM layers
        self.bilstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1,
                               batch_first=True, bidirectional=True)  # BiLSTM1
        self.bilstm2 = nn.LSTM(input_size=128, hidden_size=32, num_layers=1,
                               batch_first=True, bidirectional=True)  # BiLSTM2

        # Bottleneck layer
        self.fc_latent = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.conv1(x)  # First convolution
        x = self.leaky_relu(x)
        x = self.conv2(x)  # Second convolution
        x = self.leaky_relu(x)
        x = self.pool(x)  # Pooling

        # Flatten for LSTM input
        x = x.squeeze(dim=2).transpose(1, 2)  # Shape: (batch, time, features)
        x, _ = self.bilstm1(x)  # BiLSTM1
        x, _ = self.bilstm2(x)  # BiLSTM2

        # Reshape back for additional convolution
        x = x.unsqueeze(1)  # Add channel dimension


        # Latent representation
        x = self.fc_latent(x[:, :, -1, :].view(x.size(0), -1))
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_size):
        super(Decoder, self).__init__()

        # Dense layer
        self.fc_decoder = nn.Linear(latent_dim, 64)

        # Upsampling and DeConv2D layers
        self.up_layer = nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False)
        self.deconv_layer = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 5), padding=(0, 2))

        # Additional Transpose CNN to meet the output requirement
        self.conv_final1 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(1, 5), padding=(0, 2))
        self.conv_final2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(1, 5), padding=(0, 2))
        self.linear_final = nn.Linear(128, input_size)

        self.pooling = 200  # Output sequence length

    def forward(self, features):
        x = self.fc_decoder(features)  # Dense layer

        x = x.unsqueeze(-1).unsqueeze(-1)  # Expand dimensions
        upsampled = self.up_layer(x)  # Upsample
        x = self.deconv_layer(upsampled)[:, :, :, :self.pooling].contiguous()  # DeConv2D

        x = self.conv_final1(x)  # Additional transpose convolution layer 1
        x = self.conv_final2(x)  # Additional transpose convolution layer 2
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)  # Reshape to (batch, seq_len, channels)
        x = self.linear_final(x)
        return x.unsqueeze(1)

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, latent_dim, input_size = 200):
        super(Autoencoder, self).__init__()
        torch.manual_seed(12231)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, input_size)

    def forward(self, x):
        latent = self.encoder(x)  # Encode input to latent representation
        output = self.decoder(latent)  # Decode latent representation to reconstruct input
        return latent, output