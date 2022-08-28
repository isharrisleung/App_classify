#coding = 'utf-8'
import fasttext
import numpy as np
import sklearn.metrics as metrics
import time
import csv

data_path = 'D:/lu/fasttext_data/row_without_stop_word/' #应用类型识别挑战赛公开数据/
nb_classes = 19
train_file = "train"
test_file = "dev"

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

name_stop_word=[99,110,98,105,102]
des_stop_word=[15252363,14720387,14785469,14717871,14976901]
def deal_data(test_file):
    data_path="D:/lu/fasttext_data/"
    des_data=[]
    name_data=[]
    lbs=[]
    with open(data_path +"description/"+test_file+ ".txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i,line in enumerate(lines[:]):
            label = line.strip().split(' ')[0]
            lbs.append(label)
            des=line.strip().split(' ')[1:]
            txt=[]
            for word in des:
                if int(word) not in des_stop_word:
                    txt.append(word)
            des_data.append(txt)

    with open(data_path +"name/"+test_file+ ".txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i,line in enumerate(lines[:]):
            label = line.strip().split(' ')[0]
            if label != lbs[i]:
                print("False")
            des=line.strip()

model = fasttext.train_supervised(data_path + train_file+".txt", dim=dim, epoch=epoch, lr=lr, wordNgrams=wordngrams, verbose=2, minCount=minCount, label_prefix="__label__")
model.save_model("fasttext_model.bin",)
classifier = fasttext.load_model("fasttext_model.bin")