import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder class
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)  # Conv1D
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)  # Leaky ReLU
        self.pool = nn.MaxPool1d(kernel_size=2)  # MaxPooling
        
        self.bilstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, 
                               batch_first=True, bidirectional=True)  # BiLSTM1
        self.bilstm2 = nn.LSTM(input_size=128, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True)  # BiLSTM2
        
        self.fc_latent = nn.Linear(64, latent_dim)  # Bottleneck layer

    def forward(self, x):
        x = self.conv1(x)  # Switch to (batch, channels, time) for Conv1D
        x = self.leaky_relu(x)
        x = self.pool(x)
        
        x = x.transpose(1, 2)  # Switch to (batch, time, features) for LSTM
        x, _ = self.bilstm1(x)  # BiLSTM1
        x, _ = self.bilstm2(x)  # BiLSTM2
        
        x = self.fc_latent(x[:, -1, :])  # Bottleneck, use the last time step
        return x


# Decoder class
class Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Decoder, self).__init__()
        
        self.fc_decoder = nn.Linear(input_dim, 64)  # Dense layer
        self.bilstm_decoder = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, 
                                       batch_first=True, bidirectional=True)  # BiLSTM Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)  # Upsampling
        self.deconv = nn.Conv1d(in_channels=128, out_channels=2, kernel_size=3, padding=1)  # DeConv1D

    def forward(self, x):
        x = self.fc_decoder(x)  # Dense layer
        x = x.unsqueeze(1).repeat(1, 100, 1)  # Repeat vector to match time steps
        
        x, _ = self.bilstm_decoder(x)  # BiLSTM Decoder
        x = self.upsample(x.transpose(1, 2))  # Upsample and switch to (batch, channels, time)
        x = self.deconv(x)  # DeConv1D and switch back to (batch, time, features)
        return x


# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        torch.manual_seed(12231)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = x.squeeze(axis = 1)
        latent = self.encoder(x)  # Encode input to latent representation
        output = self.decoder(latent)  # Decode latent representation to reconstruct input
        output = output.unsqueeze(axis = 1)
        return latent, output

