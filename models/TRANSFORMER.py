import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from .BasicModule import BasicModule

class TRANSFORMER(BasicModule):
    def __init__(self, args, vectors=None):
        super(TRANSFORMER, self).__init__()

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

        self.name_att = nn.TransformerEncoder(nn.TransformerEncoderLayer(args.embedding_dim, args.attn_head, self.hidden_dim), num_layers=args.transformer_layer_num)
        self.desc_att = nn.TransformerEncoder(nn.TransformerEncoderLayer(args.embedding_dim, args.attn_head, self.hidden_dim), num_layers=args.transformer_layer_num)

        self.fc1 = nn.Linear(args.embedding_dim * 2, self.hidden_dim)
        self.dp1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(self.hidden_dim, args.label_size)

    def forward(self, name, name_length, name_mask, desc, desc_length, desc_mask):
        name = self.embedding(name).permute(1, 0, 2)
        name = self.name_att(name, src_key_padding_mask=name_mask).permute(1, 0, 2)
        name = torch.mean(name, dim=1)
        desc = self.embedding(desc).permute(1, 0, 2)
        desc = self.desc_att(desc, src_key_padding_mask=desc_mask).permute(1, 0, 2)
        desc = torch.mean(desc, dim=1)

        feat = torch.cat([name, desc], dim=1)
        y = self.fc1(feat)
        y = self.dp1(y)
        y = self.fc2(y)
        return y