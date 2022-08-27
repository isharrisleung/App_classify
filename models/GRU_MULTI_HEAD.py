import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

# class ScaledDotProductAttention(nn.Module):
#     """Scaled dot-product attention mechanism."""

#     def __init__(self, attention_dropout=0.0):
#         super(ScaledDotProductAttention, self).__init__()
#         self.dropout = nn.Dropout(attention_dropout)
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, q, k, v, scale=None, attn_mask=None):
#         """
#         前向传播.
#         Args:
#         	q: Queries张量，形状为[B, L_q, D_q]
#         	k: Keys张量，形状为[B, L_k, D_k]
#         	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
#         	scale: 缩放因子，一个浮点标量
#         	attn_mask: Masking张量，形状为[B, L_q, L_k]

#         Returns:
#         	上下文张量和attention张量
#         """
#         attention = torch.bmm(q, k.transpose(1, 2))
#         if scale:
#             attention = attention * scale
#         if attn_mask:
#             # 给需要 mask 的地方设置一个负无穷
#             attention = attention.masked_fill_(attn_mask, -np.inf)
# 	# 计算softmax
#         attention = self.softmax(attention)
# 	# 添加dropout
#         attention = self.dropout(attention)
# 	# 和V做点积
#         context = torch.bmm(attention, v)
#         return context, attention

# class MultiHeadAttention(nn.Module):

#     def __init__(self, model_dim=512, num_heads=8, dropout=0.2):
#         super(MultiHeadAttention, self).__init__()

#         self.dim_per_head = model_dim // num_heads
#         self.num_heads = num_heads
#         self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
#         self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
#         self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

#         self.dot_product_attention = ScaledDotProductAttention(dropout)
#         self.linear_final = nn.Linear(model_dim, model_dim)
#         self.dropout = nn.Dropout(dropout)
	
#         # multi-head attention之后需要做layer norm
#         self.layer_norm = nn.LayerNorm(model_dim)

#     def forward(self, key, value, query, attn_mask=None):
# 	# 残差连接
#         residual = query
#         dim_per_head = self.dim_per_head
#         num_heads = self.num_heads
#         batch_size = key.size(0)

#         # linear projection
#         key = self.linear_k(key)
#         value = self.linear_v(value)
#         query = self.linear_q(query)

#         # split by heads
#         key = key.view(batch_size * num_heads, -1, dim_per_head)
#         value = value.view(batch_size * num_heads, -1, dim_per_head)
#         query = query.view(batch_size * num_heads, -1, dim_per_head)

#         if attn_mask:
#             attn_mask = attn_mask.repeat(num_heads, 1, 1)

#         # scaled dot product attention
#         scale = (key.size(-1)) ** -0.5
#         context, attention = self.dot_product_attention(
#           query, key, value, scale, attn_mask)

#         # concat heads
#         context = context.view(batch_size, -1, dim_per_head * num_heads)

#         # final linear projection
#         output = self.linear_final(context)

#         # dropout
#         output = self.dropout(output)

#         # add residual and norm layer
#         output = self.layer_norm(residual + output)

#         return output

class GRU_MULTI_HEAD(nn.Module):
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