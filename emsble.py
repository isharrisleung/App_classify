import pandas as pd

all_test = pd.read_csv('./data/sample_submit.csv')
all_test["new_label"] = ""
a = pd.read_csv("./submit4_fasttext_lr0.8_ep9_wn2_mc2_74.358.csv")["label"]
b = pd.read_csv("./ckp/Fasttext_67851/Fasttext_0.6483909415971395.csv")["label"]
c = pd.read_csv("./ckp/GRU_MULTI_HEAD/GRU_MULTI_HEAD_0.67183.csv")["label"]
d = pd.read_csv("./ckp/GRU/GRU_0.66111.csv")["label"]

all_test["1"] = a
all_test["2"] = b
all_test["3"] = c
all_test["4"] = d

for i in range(len(all_test)):
    tmp = list(all_test.iloc[i, 3:].values)
    all_test.loc[i, "new_label"] = max(set(tmp),key=tmp.count)

all_test["label"] = all_test["new_label"]
all_test[['id', 'label']].to_csv('{}.csv'.format("all_test"), index=None)

