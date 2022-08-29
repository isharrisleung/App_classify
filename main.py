import torch
import torch.nn.functional as F
import models
import data
from config import DefaultConfig
import pandas as pd
import os
from sklearn import metrics
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from gensim.models.word2vec import Word2Vec, KeyedVectors
from tqdm import tqdm
import sys
import torch.nn as nn
import gc
from collections import deque
# from sklearn.model_selection import train_test_split
import fire

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

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def get_optimizer(model_params, lr1, lr2=0, weight_decay=0):
        optimizer = torch.optim.Adam(model_params, lr=lr1, weight_decay=weight_decay)
        return optimizer

def main(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)
    seed_torch(args.seed)
    if not torch.cuda.is_available() or not args.cuda:
        args.cuda = False
        args.device = "cpu"
    else:
        args.device = "cuda:{}".format(args.device)

    print(args.print_config())
    criterion = nn.CrossEntropyLoss()
    train_data = pd.read_csv(args.data_path)
    args.save_dir = os.path.join(args.save_dir, args.model)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    embedding_model = KeyedVectors.load_word2vec_format(args.embedding_path, binary=args.binary)

    all_best_score = []

    all_test = pd.read_csv('./data/sample_submit.csv')
    all_test["new_label"] = 0
    kf = StratifiedKFold(n_splits=args.fold, random_state=args.seed, shuffle=True)
    for k, (train_idx, val_idx) in enumerate(kf.split(train_data, train_data["new_label"].values)):
        print("----------------{} fold----------------".format(k))
        train = train_data.loc[train_idx, ["name", "description", "new_label"]]
        val = train_data.loc[val_idx, ["name", "description", "new_label"]]
        train_dataset = data.AppDataset(embedding_model, train, args, "WordVec")
        val_dataset = data.AppDataset(embedding_model, val, args, "WordVec")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_work, pin_memory=False, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_work, pin_memory=False, drop_last=False)

        # init
        if not args.use_embed:
            print("不使用预训练词向量")
            model = getattr(models, args.model)(args)
        else:
            model = getattr(models, args.model)(args, embedding_model.vectors)
        model.to(args.device)
        optimizer = model.get_optimizer(args.lr1, args.lr2, args.weight_decay)
        # optimizer = get_optimizer(model.parameters(), args.lr1, args.lr2, args.weight_decay)
        best_model_path, best_score = train_model(model, optimizer, criterion, args.lr1, args.lr2, train_dataloader, val_dataloader, args, k)
        all_best_score.append(best_score)

        ckp = torch.load(best_model_path, map_location=args.device)
        model.load_state_dict(ckp["model_state_dict"])
        test_data = pd.read_csv(args.test_path)
        test_dataset = data.TestAppDataset(embedding_model, test_data, args, "WordVec")
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_work, pin_memory=False, drop_last=False)
        test_pred = test(model, test_dataloader, args)
        all_test["{}_label".format(k)] = test_pred["new_label"]
        del model
        del optimizer
        del train_dataloader
        del val_dataloader
        del test_dataloader
        gc.collect()

    for i in range(len(all_test)):
        tmp = list(all_test.iloc[i, 3:].values)
        all_test.loc[i, "new_label"] = max(set(tmp),key=tmp.count)
    all_test["label"] = all_test.new_label.apply(transform_label)
    result_path = args.save_dir + '/{}_{}'.format(args.model, round(sum(all_best_score)/len(all_best_score), 5))
    all_test[['id', 'label']].to_csv('{}.csv'.format(result_path), index=None)
    print('Result {}.csv saved!'.format(result_path))

def  save_checkpoint_state(epoch,model,optimizer,path, score):
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
                }   
    # if not os.path.isdir(path):
    #     os.mkdir(path)

    torch.save(checkpoint, path)

def train_model(model, opt, loss_func, lr1, lr2, train_data, val_data, args, k):
    model.train()
    patient = 0
    loss_li = []
    model_li = deque()
    best_model_path = ""
    args.best_score = -10
    for e in range(args.max_epochs):
        model.train()
        databar = tqdm(train_data, file=sys.stdout)
        for idx, data in enumerate(databar):
            opt.zero_grad()
            # if idx == 13:
            #     print(app_desc)
            #     print(len_desc)
            #     # print(mask_name)
            #     input()
            app_name, len_name, mask_name, app_desc, len_desc, mask_desc, label = read_data(data, args)
            pred = model(app_name, len_name, mask_name, app_desc, len_desc, mask_desc)
            label = label.flatten()
            loss = loss_func(pred, label)
            loss.backward()
            opt.step()
            loss_li.append(loss.detach().cpu().numpy().item())
            avg_loss = np.round(np.mean(loss_li), 4)
            # print(loss_li)
            # if avg_loss == np.nan:
            #     print(avg_loss)
            databar.set_description(f"Epoch {e + 1} Loss: {avg_loss}")

        score = val(model, val_data, args)
        args.best_score = max(score, args.best_score)
        if score == args.best_score:
            patient = 0
            output_model_file = f"{args.save_dir}/model_k_{k}_epoch_{e+1}_score_{np.round(score, 5)}.bin"
            model_li.append(output_model_file)
            if len(model_li) > args.save_model_num:
                tmp_path = model_li.popleft()
                os.remove(tmp_path)
            best_model_path = output_model_file
            save_checkpoint_state(e+1, model, opt, output_model_file, score)
        else:
            patient += 1
            if patient >= args.patient:
                break
        # else:
        #     # embed参数分开训练
        #     model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
        #     lr1 *= args.lr_decay
        #     lr2 = 2e-4 if lr2 == 0 else lr2 * 0.8
        #     opt = model.get_optimizer(lr1, lr2, 0)
        #     if lr1 < args.min_lr:
        #         break
    return best_model_path, args.best_score


def test(model, test_data, args):
    # 生成测试提交数据csv
    # 将模型设为验证模式
    model.eval()

    result = np.zeros((0,))
    probs_list = []
    with torch.no_grad():
        for batch in test_data:
            app_name, len_name, mask_name, app_desc, len_desc, mask_desc = read_data(batch, args)
            outputs = model(app_name, len_name, mask_name, app_desc, len_desc, mask_desc)
            probs = F.log_softmax(outputs, dim=1)
            # probs_list.append(probs.cpu().numpy())
            pred = np.argmax(probs.cpu(), axis=1).flatten()
            result = np.hstack((result, pred.cpu().numpy()))

    # 生成概率文件npy
    # prob_cat = np.concatenate(probs_list, axis=0)

    test = pd.read_csv('./data/sample_submit.csv')
    test_id = test['id'].copy()
    test_pred = pd.DataFrame({'id': test_id, 'new_label': result})
    # test_pred['class'] = (test_pred['class'] + 1).astype(int)

    return test_pred


def val(model, dataset, args):
    model.eval()

    acc_n = 0
    val_n = 0
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    with torch.no_grad():
        for batch in dataset:
            app_name, len_name, mask_name, app_desc, len_desc, mask_desc, label = read_data(batch, args)
            label = label.flatten().cpu().numpy()
            pred = model(app_name, len_name, mask_name, app_desc, len_desc, mask_desc)
            probs = F.log_softmax(pred, dim=1)
            pred = np.argmax(probs.cpu().numpy(), axis=1)
            acc_n += (pred == label).sum().item()
            val_n += label.shape[0]
            predict = np.hstack((predict, pred))
            gt = np.hstack((gt, label))

    acc = 100. * acc_n / val_n
    f1score = metrics.f1_score(predict, gt, average="micro")
    print('* Test Acc: {:.3f}%({}/{}), F1 Score: {}'.format(acc, acc_n, val_n, f1score))
    return f1score


if __name__ == '__main__':
    fire.Fire()