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
    
class AutoEncoder(nn.Module):
    def __init__(self, config, latent_dim, seed=220617):
        super(AutoEncoder, self).__init__()
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed(seed)

        # Encoder
        self.encoder = nn.Sequential()
        if 'conv_stack' in config['architecture'].keys():
            conv_config = config['architecture']['conv_stack']
            conv_stack = []
            in_channels = 1
            for _conv in conv_config:
                name, out_channels, kernel_size, stride = _conv
                padding = tuple([k // 2 for k in kernel_size])
                conv = nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=tuple(kernel_size), stride=tuple(stride),
                    padding=padding, bias=False
                )
                conv = nn.Sequential(
                    conv,
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(1 - config['keep_prob']),
                )
                conv_stack.append((name, conv))
                in_channels = out_channels
            self.encoder = nn.Sequential(OrderedDict(conv_stack))

        rnn_config = config['architecture']['rnn_stack']
        rnn_stack = []
        rnn_input_size = config.get('rnn_input_size', 128)
        for _rnn in rnn_config:
            name, hidden_size, batch_norm, bidirectional = _rnn
            rnn = BatchRNN(
                input_size=rnn_input_size, hidden_size=hidden_size,
                batch_norm=batch_norm, bidirectional=bidirectional,
                keep_prob=config['keep_prob']
            )
            rnn_stack.append((name, rnn))
            rnn_input_size = hidden_size
        self.encoder_rnn = nn.Sequential(OrderedDict(rnn_stack))

        self.latent = nn.Linear(rnn_input_size, latent_dim)

        # Decoder
        self.decoder_rnn = nn.Sequential(OrderedDict(reversed(rnn_stack)))
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, rnn_input_size, bias=False),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(OrderedDict(reversed(conv_stack)))

    def forward(self, x):
        # Encoder pass
        x = self.encoder(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).contiguous()  # TxNxH
        x = self.encoder_rnn(x)
        latent = self.latent(x.mean(dim=1))

        # Decoder pass
        x = self.decoder_fc(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.decoder_rnn(x)
        x = x.transpose(1, 2).contiguous()  # NxTxH -> TxNxH
        x = x.view(sizes[0], sizes[1], sizes[2], -1)  # Expand feature dimensions
        x = self.decoder_conv(x)
        return x
