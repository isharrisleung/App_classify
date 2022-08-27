from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle
from gensim.models.word2vec import Word2Vec, KeyedVectors
import torch
import numpy as np

class AppDataset(Dataset):
    def __init__(self, embed_model, raw_data, args, embed_method="WordVec"):
        super().__init__()
        self.raw_data = raw_data
        self.embed_model = embed_model
        self.vocab = self.embed_model.key_to_index
        self.vocab_size = args.vocab_size
        self.embed_method = embed_method
        self.name_length = args.name_max_text_len
        self.desc_length = args.desc_max_text_len
        self.embed_dim = args.embedding_dim

    def __getitem__(self, index):
        data = self.raw_data.iloc[index]
        name = data["name"]
        dec = data["description"]
        label = data["new_label"]

        name_vec, name_len, name_mask = self.get_embed(name, self.name_length)
        dec_vec, dec_len, dec_mask = self.get_embed(dec, self.desc_length)

        return name_vec, name_len, name_mask, dec_vec, dec_len, dec_mask, torch.LongTensor([label])
    
    def get_embed(self, text, length):
        vec = np.zeros((1, length))[0]
        attn_mask = np.zeros((1, length))[0]
        text_li = text.split()
        count = 0 # 记录句子长度
        min_len = min(length, len(text_li))
        for i in range(min_len):
            try:
                vec[i] = self.vocab[text_li[i]]
                count += 1
            except:
                # 如果不存在该单词
                continue
        if count < length:
            vec[count:] = self.vocab_size
            attn_mask[count:] = 1

        return torch.LongTensor(vec), count, torch.LongTensor(attn_mask).eq(1)

    def __len__(self):
        return len(self.raw_data)

class TestAppDataset(Dataset):
    def __init__(self, embed_model, raw_data, args, embed_method="WordVec"):
        super().__init__()
        self.raw_data = raw_data
        self.embed_model = embed_model
        self.vocab = self.embed_model.key_to_index
        self.vocab_size = args.vocab_size
        self.embed_method = embed_method
        self.name_length = args.name_max_text_len
        self.desc_length = args.desc_max_text_len
        self.embed_dim = args.embedding_dim

    def __getitem__(self, index):
        data = self.raw_data.iloc[index]
        name = data["name"]
        dec = data["description"]

        name_vec, name_len, name_mask = self.get_embed(name, self.name_length)
        dec_vec, dec_len, dec_mask = self.get_embed(dec, self.desc_length)

        return name_vec, name_len, name_mask, dec_vec, dec_len, dec_mask
    
    def get_embed(self, text, length):
        vec = np.zeros((1, length))[0]
        attn_mask = np.zeros((1, length))[0]
        text_li = text.split()
        count = 0 
        min_len = min(length, len(text_li))
        for i in range(min_len):
            try:
                vec[i] = self.vocab[text_li[i]]
                count += 1
            except:
                continue
        if count < length:
            vec[count:] = self.vocab_size
            attn_mask[count:] = 1
        return torch.LongTensor(vec), count, torch.LongTensor(attn_mask).eq(1)

    def __len__(self):
        return len(self.raw_data)