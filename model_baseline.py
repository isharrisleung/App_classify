# 导入相关库
import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

MAX_DF = 0.8
# 数据读取转换
train = pd.read_csv('./data/train.csv')
lb = LabelEncoder()
train['label'] = lb.fit_transform(train['label'])

# 避免出现类别3只存在于验证集的情况
tmp = pd.DataFrame(np.repeat(train[train['label']==3].values, 1, axis=0))
tmp.columns = ['id', 'name', 'description', 'label']
train = pd.concat([train, tmp]).reset_index(drop=True)
train['label'] = train['label'].astype('int')

test = pd.read_csv('./data/test.csv')
test['id'] += 10000
data = pd.concat([train, test]).reset_index(drop=True)

# 构造name + description
data['text'] = data['name'] + data['description']
# data.head()

# tfidf
title_tfidf_vector = TfidfVectorizer(max_df=MAX_DF, ngram_range=(1,2)).fit(
    data['name'].tolist())
with open("name_pickle", "wb") as f:
    pickle.dump(title_tfidf_vector, f)
desc_tfidf_vector = TfidfVectorizer(max_df=MAX_DF, ngram_range=(1,2)).fit(
    data['description'].tolist())
with open("desc_pkl", "wb") as f:
    pickle.dump(desc_tfidf_vector, f)
total_tfidf_vector = TfidfVectorizer(max_df=MAX_DF, ngram_range=(1,2)).fit(
    data['text'].tolist())

def create_csr_mat_input(title_list, desc_list, total_list):
    return sp.hstack((title_tfidf_vector.transform(title_list),
                      desc_tfidf_vector.transform(desc_list),
                      total_tfidf_vector.transform(total_list),
                      ),
                     format='csr')

tfidf_input = create_csr_mat_input(data['name'], data['description'], data['text'])

# 模型训练与预测
def train_model(X_train, X_test, features, y, seed=2021, save_model=False):

    KF = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    oof_lgb = np.zeros((X_train.shape[0], 19))
    predictions_lgb = np.zeros((X_test.shape[0], 19))

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train, y.values)):

        clf = SGDClassifier(random_state=1017, loss='log')
        # clf = LogisticRegression(random_state=1017)
        clf.fit(X_train[trn_idx], y.iloc[trn_idx])
        oof_lgb[val_idx] = clf._predict_proba_lr(X_train[val_idx])
        predictions_lgb += clf._predict_proba_lr(X_test) / 5

    print("F1 score micro: {}".format(f1_score(y, np.argmax(oof_lgb, axis=1), average='micro')))
    print("F1 score macro: {}".format(f1_score(y, np.argmax(oof_lgb, axis=1), average='macro')))
    return oof_lgb, predictions_lgb

train = data[~data['label'].isna()].reset_index(drop=True)
test = data[data['label'].isna()].reset_index(drop=True)
y = train['label']

train_len = train.shape[0]
test_len = test.shape[0]
features = [i for i in train.columns if i not in ['id', 'name', 'description', 'label', 'text']]
seeds = [2021]
pred = []
for seed in seeds:
    oof_lgb, predictions_lgb = train_model(tfidf_input[:train_len], tfidf_input[train_len:], features, y, seed)
    pred.append(predictions_lgb)

# 生成提交文件
test['label'] = np.argmax(np.mean(pred, axis=0), axis=1)
test['label'] = lb.inverse_transform(test['label'])
test['id'] -= 10000
test[['id', 'label']].to_csv('./ckp/baseline/sub_base_new_1.csv', index=False)
test[['id', 'label']].head()