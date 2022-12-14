import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from .BasicModule import BasicModule

class GRU_MULTI_HEAD(BasicModule):
    def __init__(self, args, vectors=None):
        super(GRU_MULTI_HEAD, self).__init__()

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
        
        self.desc_att = nn.MultiheadAttention(self.hidden_dim, args.attn_head)
        self.desc_bigru = nn.GRU(args.embedding_dim, self.hidden_dim // 2, num_layers=self.gru_layers, bidirectional=args.bidirectional, batch_first=True)
        self.name_att = nn.MultiheadAttention(self.hidden_dim, args.attn_head)
        self.name_bigru = nn.GRU(args.embedding_dim, self.hidden_dim // 2, num_layers=self.gru_layers, bidirectional=args.bidirectional, batch_first=True)

        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dp1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(self.hidden_dim, args.label_size)

    def forward(self, name, name_length, name_mask, desc, desc_length, desc_mask):
        name = self.embedding(name)
        name_pack = pack_padded_sequence(input=name, lengths=name_length, batch_first=True, enforce_sorted=False)
        name, _ = self.name_bigru(name_pack)
        name, _ = pad_packed_sequence(name, batch_first=True, total_length=self.name_length)
        name = name.permute(1, 0, 2)
        name, _ = self.name_att(name, name, name, key_padding_mask=name_mask)
        name = torch.mean(name.permute(1, 0, 2), dim=1)

        desc = self.embedding(desc)
        desc_pack = pack_padded_sequence(input=desc, lengths=desc_length, batch_first=True, enforce_sorted=False)
        desc, _ = self.desc_bigru(desc_pack)
        desc, _ = pad_packed_sequence(desc, batch_first=True, total_length=self.desc_legnth)
        desc = desc.permute(1, 0, 2)
        desc, _ = self.desc_att(desc, desc, desc, key_padding_mask=desc_mask)
        desc = torch.mean(desc.permute(1, 0, 2), dim=1)

        feat = torch.cat([name, desc], dim=1)
        y = self.fc1(feat)
        y = self.dp1(y)
        y = self.fc2(y)
        return y