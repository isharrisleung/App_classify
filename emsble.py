from importlib.resources import path
import pandas as pd

all_test = pd.read_csv('./data/sample_submit.csv')
all_test["new_label"] = ""
path = "./ckp/res"
import os
dirlist = os.listdir(path)
for idx, sub_path in enumerate(dirlist):
    all_test[idx] = pd.read_csv(os.path.join(path, sub_path))["label"]
# a = pd.read_csv("../submit4_fasttext_lr0.8_ep9_wn2_mc2_74.358.csv")["label"]
# b = pd.read_csv("./GRU_MULTI_HEAD_0.67183.csv")["label"]
# c = pd.read_csv("./Transformer_TextCNN_0.67587.csv")["label"]
# # d = pd.read_csv("./ckp/GRU/GRU_0.66111.csv")["label"]

# all_test["1"] = a
# all_test["2"] = b
# all_test["3"] = c
# # all_test["4"] = d

for i in range(len(all_test)):
    tmp = list(all_test.iloc[i, 3:].values)
    all_test.loc[i, "new_label"] = max(set(tmp),key=tmp.count)

all_test["label"] = all_test["new_label"]
all_test[['id', 'label']].to_csv('./ckp/{}.csv'.format("all_test"), index=None)
