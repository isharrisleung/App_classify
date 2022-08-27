from operator import concat
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Fasttext(torch.nn.Module):
    def __init__(self, args, vectors=None):
        super(Fasttext, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.embedding = nn.Embedding(args.vocab_size+1, args.embedding_dim)
        if vectors is not None:
            vectors =  np.row_stack((vectors, np.zeros(args.embedding_dim)))
            self.embedding.weight.data.copy_(torch.Tensor(vectors))

        self.fc1 = nn.Linear(args.embedding_dim * 2, self.hidden_dim)
        self.dp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(self.hidden_dim, args.label_size)

    def forward(self, name, name_length, name_mask, desc, desc_length, desc_mask):
        name = self.embedding(name)
        name = torch.sum(name, dim=1) / name_length.unsqueeze(1)

        desc = self.embedding(desc)
        desc = torch.sum(desc, dim=1) / desc_length.unsqueeze(1)

        feat = torch.cat([name, desc], dim=1)
        y = self.fc1(feat)
        y = self.dp1(y)
        y = self.fc2(y)
        return y