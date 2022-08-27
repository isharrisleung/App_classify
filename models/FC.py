from operator import concat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FC(torch.nn.Module):
    def __init__(self, args):
        super(FC, self).__init__()
        self.args = args

        self.name_length = args.name_max_text_len
        self.desc_legnth = args.desc_max_text_len

        self.hidden_dim = args.hidden_dim
        self.gru_layers = args.lstm_layers
        
        self.name_bigru = nn.GRU(args.embedding_dim, self.hidden_dim // 2, num_layers=self.gru_layers, bidirectional=args.bidirectional, batch_first=True)
        self.name_weight_W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.name_weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim, 1))

        self.desc_bigru = nn.GRU(args.embedding_dim, self.hidden_dim // 2, num_layers=self.gru_layers, bidirectional=args.bidirectional, batch_first=True)
        self.desc_weight_W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.desc_weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim, 1))

        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, args.label_size)

        nn.init.uniform_(self.name_weight_W, -0.1, 0.1)
        nn.init.uniform_(self.name_weight_proj, -0.1, 0.1)
        nn.init.uniform_(self.desc_weight_W, -0.1, 0.1)
        nn.init.uniform_(self.desc_weight_proj, -0.1, 0.1)

    def forward(self, name, name_length, desc, desc_length):
        # embeds = self.embedding(sentence) # [seq_len, bs, emb_dim]
        name_pack = pack_padded_sequence(input=name, lengths=name_length, batch_first=True, enforce_sorted=False)
        name, _ = self.name_bigru(name_pack)
        name, _ = pad_packed_sequence(name, batch_first=True, total_length=self.name_length)
        u = torch.tanh(torch.matmul(name, self.name_weight_W))
        att = torch.matmul(u, self.name_weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = name * att_score
        name = torch.sum(scored_x, dim=1)

        desc_pack = pack_padded_sequence(input=desc, lengths=desc_length, batch_first=True, enforce_sorted=False)
        desc, _ = self.desc_bigru(desc_pack)
        desc, _ = pad_packed_sequence(desc, batch_first=True, total_length=self.desc_legnth)
        u = torch.tanh(torch.matmul(desc, self.desc_weight_W))
        att = torch.matmul(u, self.desc_weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = desc * att_score
        desc = torch.sum(scored_x, dim=1)

        feat = torch.cat([name, desc], dim=1)
        y = self.fc1(feat)
        y = self.fc2(y)
        return y