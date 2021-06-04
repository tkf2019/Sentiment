import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CNN(nn.Module):
    input_size: int

    def __init__(self, input_size: int, pad_len: int,
                 hidden_size: int = 256, dropout_rate: float = 0.5) -> None:
        """
        Args:
            input_size (int): size of each word vector
            pad_len (int): padding length for each sentence in datasets
            hidden_size (int): argument hidden size of out channels of nn.Conv2d
            dropout_rate (float): argument dropout rate of dropout layer
        """
        super(CNN, self).__init__()
        self.in_channels = 1
        self.out_channels = hidden_size
        self.kernal_sizes = [2, 3, 4]
        self.dropout = dropout_rate
        self.num_classes = 7

        self.convs = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels, (kernal_size, input_size))
                                   for kernal_size in self.kernal_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(len(self.kernal_sizes) * self.out_channels,
                            self.num_classes)

    def conv_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, _):
        """
        Args:
            x (tensor): batch input with tensor [bs * padding length * size of word vector]
        """
        # print(x.size(), x.shape)
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_pool(out, conv)
                        for conv in self.convs], dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class RNN(nn.Module):
    """RNN model using bidirectional LSTM"""

    def __init__(self, input_size: int, pad_len: int,
                 hidden_size: int = 128, dropout_rate: float = 0.5) -> None:
        """
        Args:
            input_size (int): size of each word vector
            pad_len (int): padding length for each sentence in datasets
            hidden_size (int): argument hidden size of nn.LSTM model
            dropout_rate (float): argument dropout rate of nn.LSTM model
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.dropou_rate = dropout_rate
        self.bidirectional = True
        self.num_classes = 7

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True,
                            dropout=self.dropou_rate)
        self.fc = nn.Linear(self.hidden_size * 4, self.num_classes)

    def forward(self, x, x_len):
        """
        Args: 
            x (tensor): batch input with tensor [bs * padding length * size of word vector]
            x_len (int): original length of input sentence before cutting off
        """
        # print(x.size(), x.shape)
        _, idx_sort = torch.sort(x_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x = x.index_select(0, Variable(idx_sort))
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=list(x_len[idx_sort]),
                                              batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(x)
        hn = hn.permute(1, 0, 2)
        hn = hn.index_select(0, Variable(idx_unsort))
        hn = hn.reshape(hn.shape[0], -1)
        out = self.fc(hn)
        return out


class MLP(nn.Module):
    def __init__(self, input_size: int, pad_len: int,
                 hidden_size: int = 512, dropout_rate: int = 0.3) -> None:
        """
        Args:
            input_size (int): size of each word vector
            pad_len (int): padding length for each sentence in datasets
            hidden_size (int): argument hidden size of hidden layer
            dropout_rate (float): argument dropout rate of dropout layer
        """
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_classes = 7

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(input_size * pad_len, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, _):
        """
        Args: 
            x (tensor): batch input with tensor [bs * padding length * size of word vector]
        """
        # print(x.size(), x.shape)
        out = x.view(x.shape[0], -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out
