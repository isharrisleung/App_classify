# 导入相关库
import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

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
test["new_label"] = ""
data = pd.concat([train, test]).reset_index(drop=True)

# 构造name + description
data['text'] = data['name'] + data['description']
# data.head()

# tfidf
title_tfidf_vector = TfidfVectorizer(max_df=MAX_DF, ngram_range=(1,2)).fit(
    data['name'].tolist())
desc_tfidf_vector = TfidfVectorizer(max_df=MAX_DF, ngram_range=(1,2)).fit(
    data['description'].tolist())
total_tfidf_vector = TfidfVectorizer(max_df=MAX_DF, ngram_range=(1,2)).fit(
    data['text'].tolist())

def create_csr_mat_input(title_list, desc_list, total_list):
    return sp.hstack((title_tfidf_vector.transform(title_list),
                      desc_tfidf_vector.transform(desc_list),
                      ),
                     format='csr')

tfidf_input = create_csr_mat_input(data['name'], data['description'], data['text'])

# 模型训练与预测
def train_model(X_train, X_test, features, y, seed=2021, save_model=False):

    KF = StratifiedKFold(n_splits=2, random_state=seed, shuffle=True)
    oof_lgb = np.zeros((X_train.shape[0],))
    # predictions_lgb = np.zeros((2, X_test.shape[0]))

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train, y.values)):
        print("fold ", fold_)
        clf = svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovo', random_state=1017)
        # clf = SGDClassifier(random_state=1017, loss='log')
        # clf = LogisticRegression(random_state=1017)
        print(y.iloc[trn_idx])
        clf.fit(X_train[trn_idx], y.iloc[trn_idx])
        oof_lgb[val_idx] = clf.predict(X_train[val_idx])
        # predictions_lgb[fold_] = clf.predict(X_test)
        test["label_{}".format(fold_)] = clf.predict(X_test)
        # print(oof_lgb)
    


    print("F1 score micro: {}".format(f1_score(y, oof_lgb, average='micro')))
    print("F1 score macro: {}".format(f1_score(y, oof_lgb, average='macro')))
    return oof_lgb

train = data[~data['label'].isna()].reset_index(drop=True)
test = data[data['label'].isna()].reset_index(drop=True)
y = train['label']

train_len = train.shape[0]
test_len = test.shape[0]
features = [i for i in train.columns if i not in ['id', 'name', 'description', 'label', 'text']]
seeds = [2021]
pred = []
for seed in seeds:
    oof_lgb = train_model(tfidf_input[:train_len], tfidf_input[train_len:], features, y, seed)
    # pred.append(predictions_lgb)
# print(test.info())
for i in range(len(test)):
    tmp = list(test.iloc[i, 6:].values)
    test.loc[i, "new_label"] = max(set(tmp),key=tmp.count)

# 生成提交文件
# test['label'] = np.argmax(np.mean(pred, axis=0), axis=1)
print(test['new_label'])
test['label'] = lb.inverse_transform(test['new_label'])
test['id'] -= 10000
test[['id', 'label']].to_csv('./ckp/baseline/sub_base_svm_new.csv', index=False)
test[['id', 'label']].head()