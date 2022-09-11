import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from .BasicModule import BasicModule


kernal_sizes = [1, 2, 3, 4, 5]

class Transformer_TextCNN_itdf(BasicModule):
    def __init__(self, args, vectors=None):
        super(Transformer_TextCNN_itdf, self).__init__()

        self.args = args

        self.name_length = args.name_max_text_len
        self.desc_legnth = args.desc_max_text_len

        self.hidden_dim = args.hidden_dim
        self.gru_layers = args.lstm_layers

        self.fc1 = nn.Linear(args.linear_hidden_size * 2, args.linear_hidden_size)
        self.dp1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(args.linear_hidden_size, args.label_size)
        # print(self)

    def forward(self, name, name_length, name_mask, desc, desc_length, desc_mask):
        name = self.embedding(name).permute(1, 0, 2) # 输出 seq len, batch, emb
        name = self.name_encoder(name, src_key_padding_mask=name_mask).permute(1, 2, 0) 
        name_conv_out = [conv(name) for conv in self.name_convs]
        name_conv_out = torch.cat(name_conv_out, dim=1)
        name_fcout = self.name_fc(name_conv_out.view(name_conv_out.size(0), -1))

        desc = self.embedding(desc).permute(1, 0, 2)
        desc = self.desc_encoder(desc, src_key_padding_mask=desc_mask).permute(1, 2, 0)
        desc_conv_out = [conv(desc) for conv in self.desc_convs]
        desc_conv_out = torch.cat(desc_conv_out, dim=1)
        desc_fcout = self.desc_fc(desc_conv_out.view(desc_conv_out.size(0), -1))

        feat = torch.cat([name_fcout, desc_fcout], dim=1)
        y = self.fc1(feat)
        y = self.dp1(y)
        y = self.fc2(y)
        return y