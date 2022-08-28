import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


kernal_sizes = [1, 2, 3, 4, 5]

class Transformer_TextCNN(nn.Module):
    def __init__(self, args, vectors=None):
        super(Transformer_TextCNN, self).__init__()

        # print(vectors.shape)
        self.embedding = nn.Embedding(args.vocab_size+1, args.embedding_dim)
        if vectors is not None:
            vectors =  np.row_stack((vectors, np.zeros(args.embedding_dim)))
            self.embedding.weight.data.copy_(torch.Tensor(vectors))

        self.args = args

        self.name_length = args.name_max_text_len
        self.desc_legnth = args.desc_max_text_len

        self.hidden_dim = args.hidden_dim
        self.gru_layers = args.lstm_layers
        self.desc_att = nn.TransformerEncoderLayer(args.embedding_dim, args.attn_head, self.hidden_dim)
        self.name_att = nn.TransformerEncoderLayer(args.embedding_dim, args.attn_head, self.hidden_dim)

        name_convs = [
            nn.Sequential(
                nn.Conv1d(in_channels=args.embedding_dim,
                          out_channels=args.kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(args.kernel_num),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=args.kernel_num,
                          out_channels=args.kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(args.kernel_num),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=(args.name_max_text_len - kernel_size*2 + 2))
            )
            for kernel_size in kernal_sizes
        ]

        self.name_convs = nn.ModuleList(name_convs)

        desc_convs = [
            nn.Sequential(
                nn.Conv1d(in_channels=args.embedding_dim,
                          out_channels=args.kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(args.kernel_num),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=args.kernel_num,
                          out_channels=args.kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(args.kernel_num),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=(args.desc_max_text_len - kernel_size*2 + 2))
            )
            for kernel_size in kernal_sizes
        ]

        self.desc_convs = nn.ModuleList(desc_convs)

        self.name_fc = nn.Sequential(
            nn.Linear(5 * args.kernel_num, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, args.linear_hidden_size)
        )
        self.desc_fc = nn.Sequential(
            nn.Linear(5 * args.kernel_num, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, args.linear_hidden_size)
        )

        self.fc1 = nn.Linear(args.linear_hidden_size * 2, args.linear_hidden_size)
        self.dp1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(args.linear_hidden_size, args.label_size)
        print(self)

    def forward(self, name, name_length, name_mask, desc, desc_length, desc_mask):
        name = self.embedding(name).permute(1, 0, 2) # 输出 seq len, batch, emb
        name = self.name_att(name, src_key_padding_mask=name_mask).permute(1, 2, 0) 
        name_conv_out = [conv(name) for conv in self.name_convs]
        name_conv_out = torch.cat(name_conv_out, dim=1)
        name_fcout = self.name_fc(name_conv_out.view(name_conv_out.size(0), -1))

        desc = self.embedding(desc).permute(1, 0, 2)
        desc = self.desc_att(desc, src_key_padding_mask=desc_mask).permute(1, 2, 0)
        desc_conv_out = [conv(desc) for conv in self.desc_convs]
        desc_conv_out = torch.cat(desc_conv_out, dim=1)
        desc_fcout = self.desc_fc(desc_conv_out.view(desc_conv_out.size(0), -1))

        feat = torch.cat([name_fcout, desc_fcout], dim=1)
        y = self.fc1(feat)
        y = self.dp1(y)
        y = self.fc2(y)
        return y