# coding=utf-8

import csv
import os


def csv_to_txt(csv_path,txt_path):
    String_list=[]
    content_len_list=[]
    with open(csv_path,'r') as csvfile:
        readerCSV=csv.reader(csvfile,delimiter=',')
        for row in readerCSV:
            if row[1] != 'content' and row[1]:
                content_len_list.append(len(row[1]))
                string='./'+str(row[0])+' '+str(row[1])+'\n'
                String_list.append(string)
    content_len_list.sort()
    print(content_len_list)
    with open(txt_path,'a') as txtfile:
        txtfile.writelines(String_list)

def make_test_txt(src_path):
    filename_list=os.listdir(src_path)
    String_list=[]
    for i in filename_list:
        string='./'+str(i)+' '+'??????'+'\n'
        String_list.append(string)
    with open(src_path+'test.txt','a') as txtfile:
        txtfile.writelines(String_list)


def txt_to_csv(src_path):
    filename_result=[]
    holder_list=os.listdir(src_path)
    for i in holder_list:
        filename=i.split('_._')[1]
        dir=src_path+i
        file_list=os.listdir(dir)
        for j in file_list:
            if j.endswith('.txt'):
                with open(dir+'/'+j,'r') as txtfile:
                    txtfile.readline()
                    result=txtfile.readline().strip().replace(' ','')
                    filename_result.append((filename,result))
    headers=['name','content']
    with open('/home/dilligencer/code/比赛/ocr---AI/results_train/test.csv','w') as f:
        f_csv=csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(filename_result)



def main():
    # csv_to_txt('/home/dilligencer/code/比赛/ocr---AI/competition_final/train.csv',
    #            '/home/dilligencer/code/比赛/ocr---AI/competition_final/train.txt')
    # make_test_txt('/home/dilligencer/code/比赛/ocr---AI/competition_final/test/')
    txt_to_csv('/home/dilligencer/code/比赛/ocr---AI/results_train/incorrect/')


if __name__ == '__main__':
    main()
