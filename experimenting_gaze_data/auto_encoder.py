import torch
import torch.nn as nn
from collections import OrderedDict

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, batch_norm=True, keep_prob=0.5):
        super(BatchRNN, self).__init__()
        self.batch_norm = batch_norm
        self.bidirectional = bidirectional

        rnn_bias = False if batch_norm else True
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional=bidirectional,
                          batch_first=True,
                          bias=rnn_bias)
        self.batch_norm_op = SequenceWise(nn.BatchNorm1d(hidden_size))

        self.dropout_op = nn.Dropout(1-keep_prob)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.contiguous()
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
            x = x.contiguous()
        if self.batch_norm:
            x = self.batch_norm_op(x)
        x = self.dropout_op(x)
        return x
    import torch
import torch.nn as nn
from collections import OrderedDict

class gazeAutoEncoder(nn.Module):
    def __init__(self, latent_dim, seed=220617):
        super(gazeAutoEncoder, self).__init__()
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed(seed)

        model_architecture = {
            "conv_stack": [["conv1", 8, [2, 11], [1, 1]],
                          ],
            "rnn_stack": [["gru1", 64, True, True],
                          ["gru2", 64, True, True],
                          ]
        }

        # Encoder: Convolutional Stack
        if 'conv_stack' in model_architecture.keys():
            conv_config = model_architecture['conv_stack']
            conv_stack = []
            feat_dim = 2
            in_channels = 1
            for _conv in conv_config:
                name, out_channels, kernel_size, stride = _conv
                padding = list(map(lambda x: int(x / 2), kernel_size))
                _conv = nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=tuple(kernel_size), stride=tuple(stride),
                    padding=tuple(padding),
                    bias=False
                )
                _conv = nn.Sequential(
                    _conv,
                    nn.BatchNorm2d(out_channels),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Dropout(1 - 0.75),
                )
                conv_stack.append((name, _conv))
                in_channels = out_channels
                feat_dim = feat_dim / stride[0] + 1
            self.encoder_conv = nn.Sequential(OrderedDict(conv_stack))
            rnn_input_size = int(feat_dim * out_channels)
        else:
            self.encoder_conv = None
            rnn_input_size = 2

        # Encoder: RNN Stack
        rnn_config = model_architecture['rnn_stack']
        rnn_stack = []
        for _rnn in rnn_config:
            name, hidden_size, batch_norm, bidirectional = _rnn
            _rnn = BatchRNN(
                input_size=rnn_input_size, hidden_size=hidden_size,
                bidirectional=bidirectional, batch_norm=batch_norm,
                keep_prob=0.75
            )
            rnn_stack.append((name, _rnn))
            rnn_input_size = hidden_size
        self.encoder_rnn = nn.Sequential(OrderedDict(rnn_stack))

        # Latent Space
        self.latent = nn.Linear(hidden_size, latent_dim)

        # Decoder: Reverse RNN Stack
        decoder_rnn_stack = []
        for _rnn in reversed(rnn_config):
            name, hidden_size, batch_norm, bidirectional = _rnn
            _rnn = BatchRNN(
                input_size=24, hidden_size=24,
                bidirectional=bidirectional, batch_norm=batch_norm,
                keep_prob=0.75
            )
            decoder_rnn_stack.append((name + "_dec", _rnn))
            latent_dim = hidden_size
        self.decoder_rnn = nn.Sequential(OrderedDict(decoder_rnn_stack))

        # Decoder: Reverse Convolutional Stack
        decoder_conv_stack = []
        for _conv in reversed(conv_config):
            name, out_channels, kernel_size, stride = _conv
            padding = list(map(lambda x: int(x / 2), kernel_size))
            _conv = nn.ConvTranspose2d(
                out_channels, in_channels,
                kernel_size=tuple(kernel_size), stride=tuple(stride),
                padding=tuple(padding),
                bias=False
            )
            _conv = nn.Sequential(
                _conv,
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(1 - 0.75),
            )
            decoder_conv_stack.append((name + "_dec", _conv))
            in_channels = out_channels
        self.decoder_conv = nn.Sequential(OrderedDict(decoder_conv_stack))
        self.final_conv = nn.ConvTranspose2d(
                in_channels=8, out_channels=1, kernel_size=1, stride=1
            )

    def forward(self, in_x):
        # Encoder
        if self.encoder_conv is not None:
            x = self.encoder_conv(in_x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).contiguous()  # TxNxH
        x = self.encoder_rnn(x)

        # Latent Space
        x = self.latent(x)

        # Decoder
        x = self.decoder_rnn(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(sizes[0], sizes[1], sizes[2], sizes[3])  # Reshape to original dims
        x = self.decoder_conv(x)
        x = self.final_conv(x)
        return x
