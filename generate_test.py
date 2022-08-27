import pandas as pd
import models
from config import DefaultConfig
from gensim.models.word2vec import Word2Vec, KeyedVectors
import torch
import data
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
import gc

label_dict={0:'14783134 15697333 14854817 14925479',1:'14847385 14844587 14848641 14847398', \
     2:'14786237 15697082 14722731 14924977', 3:'14844093 15705739 14854331 15699885', \
    4: '15632285 15706536 14721977 14925219', 5:'15630486 15702410 14718849 15709093', \
    6: '14924216 14781104 14717848 14791612', 7:'14782903 15634620 15638402 15706300', \
    8: '14858934 15636660 15704193 14849963', 9:'15709098 14716590 14924703 14779559', \
     10:'14726332 14728344 14854542 14844591',11: '14856354 14844592', \
     12:'15710359 14847407 14845602 14859696', 13:'14794687 14782344', \
     14:'14925756 15639967 14853254 14728639', 15:'14844593 14924945', \
    16: '14844856 14724258 14925237 14854807', 17:'14852788 14717848 15639958 15632020', \
     18:'14784131 14858934 14784131 14845064'}

def transform_label(label):
    return label_dict[int(label)]

def read_data(data, args):
    return tuple(d.to(args.device) for d in data)
def test(model, test_data, args):
    # 生成测试提交数据csv
    # 将模型设为验证模式
    model.eval()

    result = np.zeros((0,))
    probs_list = []
    with torch.no_grad():
        for batch in test_data:
            app_name, len_name, app_desc, len_desc = read_data(batch, args)
            outputs = model(app_name, len_name, app_desc, len_desc)
            probs = F.log_softmax(outputs, dim=1)
            # probs_list.append(probs.cpu().numpy())
            pred = np.argmax(probs, axis=1).flatten()
            result = np.hstack((result, pred.cpu().numpy()))

    # 生成概率文件npy
    # prob_cat = np.concatenate(probs_list, axis=0)

    test = pd.read_csv('./data/sample_submit.csv')
    test_id = test['id'].copy()
    test_pred = pd.DataFrame({'id': test_id, 'new_label': result})
    # test_pred['class'] = (test_pred['class'] + 1).astype(int)

    return test_pred

if __name__ == "__main__":
    path = ["ckp\GRU_att_71738\model_k_0_epoch_9_score_0.66071.bin", 
            "ckp\GRU_att_71738\model_k_1_epoch_17_score_0.6619.bin", 
            "ckp\GRU_att_71738\model_k_2_epoch_15_score_0.62857.bin",
            "ckp\GRU_att_71738\model_k_3_epoch_13_score_0.68571.bin",
            "ckp\GRU_att_71738\model_k_4_epoch_15_score_0.65316.bin"]
    args = DefaultConfig()
    embedding_model = KeyedVectors.load_word2vec_format(args.embedding_path, binary=True)

    all_test = pd.read_csv('./data/sample_submit.csv')
    all_test["new_label"] = 0

    for m in range(len(path)):
        print(m)
        model = getattr(models, "GRU_att")(args, embedding_model.vectors)
        ckp = torch.load(path[m])
        model.load_state_dict(ckp["model_state_dict"])
        test_data = pd.read_csv(args.test_path)
        test_dataset = data.TestAppDataset(embedding_model, test_data, args, "WordVec")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_work, pin_memory=False, drop_last=False)
        test_pred = test(model, test_dataloader, args)
        all_test["{}_label".format(m)] = test_pred["new_label"]
        del model
        del test_dataloader
        gc.collect()


    alu_csv = pd.read_csv('./submit4_fasttext_lr0.8_ep9_wn2_mc2_74.358.csv')
    all_test["new_new_label"] = alu_csv["label"]

    for i in range(len(all_test)):
        tmp = list(all_test.iloc[i, 3:].values)
        all_test.loc[i, "new_label"] = max(set(tmp),key=tmp.count)
    all_test["label"] = all_test.new_label.apply(transform_label)

    all_test[['id', 'label']].to_csv('{}.csv'.format("all_test"), index=None)