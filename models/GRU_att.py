from operator import concat
from tabnanny import verbose
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output

class SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)   
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)

        return attn, output



class GRU_att(torch.nn.Module):
    def __init__(self, args, vectors=None):
        super(GRU_att, self).__init__()

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
        
        self.name_bigru = nn.GRU(args.embedding_dim, self.hidden_dim // 2, num_layers=self.gru_layers, bidirectional=args.bidirectional, batch_first=True)
        self.name_att = SelfAttention(n_head=args.attn_head, d_k=self.hidden_dim, d_v=self.hidden_dim // 2, d_x=self.hidden_dim, d_o=self.hidden_dim)

        self.desc_bigru = nn.GRU(args.embedding_dim, self.hidden_dim // 2, num_layers=self.gru_layers, bidirectional=args.bidirectional, batch_first=True)
        self.desc_att = SelfAttention(n_head=args.attn_head, d_k=self.hidden_dim, d_v=self.hidden_dim // 2, d_x=self.hidden_dim, d_o=self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, args.label_size)

    def forward(self, name, name_length, desc, desc_length):
        name = self.embedding(name)
        name_pack = pack_padded_sequence(input=name, lengths=name_length, batch_first=True, enforce_sorted=False)
        name, _ = self.name_bigru(name_pack)
        name, _ = pad_packed_sequence(name, batch_first=True, total_length=self.name_length)
        print(name[0])
        input()
        _, name = self.name_att(name)
        name = torch.sum(name, dim=1)

        desc = self.embedding(desc)
        desc_pack = pack_padded_sequence(input=desc, lengths=desc_length, batch_first=True, enforce_sorted=False)
        desc, _ = self.desc_bigru(desc_pack)
        desc, _ = pad_packed_sequence(desc, batch_first=True, total_length=self.desc_legnth)
        _, desc = self.desc_att(desc)
        desc = torch.sum(desc, dim=1)

        feat = torch.cat([name, desc], dim=1)
        y = self.fc1(feat)
        y = self.fc2(y)
        return y