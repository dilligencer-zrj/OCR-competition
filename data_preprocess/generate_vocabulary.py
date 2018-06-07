# coding=utf-8

import csv
import json
import os
from PIL import Image

def mywritejson(save_path,content):
    content=json.dumps(content,indent=4,ensure_ascii=False)
    with open(save_path,'w') as f:
        f.write(content)


def generate_vocabulary(src_path,output_path):
    vocabulary=set()
    with open(src_path,'r') as csvfile:
        readerCSV=csv.reader(csvfile,delimiter=',')
        for row in readerCSV:
            if row[1] != 'content':
                for word in row[1]:
                    if word not in vocabulary:
                        vocabulary.add(word)
    vocabulary=list(vocabulary)
    vocabulary_dict=dict()
    vocabulary_dict['vocabulary']=vocabulary
    mywritejson(output_path,vocabulary_dict)



def statics(src_path):
    len_list=[]
    with open(src_path,'r') as csvfile:
        readerCSV=csv.reader(csvfile,delimiter=',')
        for row in readerCSV:
            if row[1] != 'content':
                if len(row[1])==0:
                    print(row[0])
                len_list.append(len(row[1]))
        min_length=min(len_list)
        max_length=max(len_list)
        # print(len_list)
        print(min_length,max_length)

def picture_size(src_path):
    w_list=[]
    h_list=[]
    picture_list=os.listdir(src_path)
    for i in picture_list:
        if i.endswith('.png'):
            file_path=src_path+i
            with open(file_path,'rb') as img_file:
                img=Image.open(img_file)
                w,h=img.size
                aspect_ratio = float(w) / float(h)
                w=aspect_ratio*48
                w_list.append(w)
    w_list.sort()
    print(w_list[-63])





def main():
    # generate_vocabulary('/home/dilligencer/code/比赛/ocr---AI/competition_final/train.csv',
    #                     '/home/dilligencer/code/比赛/ocr---AI/competition_final/vocabulary.json')
    # statics('/home/dilligencer/code/比赛/ocr---AI/competition_final/train.csv')
    picture_size('/home/dilligencer/code/比赛/ocr---AI/competition_final/train/')

if __name__ == "__main__":
    main()
