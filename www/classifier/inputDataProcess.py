import pandas as pd
import numpy as np
import jieba
from gensim.models import word2vec, KeyedVectors
from gensim import corpora
import os
from tqdm import tqdm
import re

class DataPreprocess(object):
    
    def __init__(self):
        self.data = None
        self.seg_list = []
        self.stopWords = set()
        
    def getInputData(self, path='./data_LC/', name="test.tsv"):
        print("Reading TestData")
        self.data = pd.read_table(path + name,
                                  encoding="gb18030",
                                  quoting=3,
                                  header = None,
                                  names = ['ITEM_NAME']
                                  )
        return self.data
    
    def _stopwordslist(self, stopwordpath):
        print("insert stopwords")
        stopwords = [line.strip() for line in open(stopwordpath, 'r', encoding='gb18030').readlines()]
        self.stopWords = set(stopwords)
        
    def _suggestFreq(self):
        freq = ['腾讯QQ', 'q币', 'qq币', 'Q币', 'QQ币', 'QB', 'qb', 'DVD','黄钻','红钻','黑钻','绿钻','狗粮','成犬'
               ,'狗盆','猫碗','狗碗','泡茶','煮水','煮茶','蓝钻','紫钻']
        for word in freq:
            jieba.suggest_freq(word, True)
            
    def outletCarveData(self, path, name, batch = False):
        if not os.path.exists(path):
            os.mkdir(path)
        saveData = pd.DataFrame(self.seg_list)
        if(batch):
            saveData.to_csv(path + name,encoding="gb18030",header = False, index = False, mode = 'a')
        else:
            saveData.to_csv(path + name,encoding="gb18030",header = False, index = False)
        del saveData
        del self.seg_list
        self.seg_list = []
        #将pandas数据存成文件

    def outletTextCarve(self,
                        stopWordPath="./data_preprocess/chineseStopWords.txt",
                        ):
        self._stopwordslist(stopWordPath)
        self._suggestFreq()
        texts = self.data
        seg = jieba.lcut(self.clean_numbers(texts), cut_all=False)
        newSeg = []
        while (len(seg)):
            # print(len(seg))
            word = seg.pop(0)
            if word != '' and word != ' ':
                if word not in self.stopWords:
                    if word != '\t' and word != '\r\n':
                        newSeg.append(word)
        listStr = " ".join(newSeg)
        if (listStr == ''):
            listStr = '无文本'
        return listStr

    def outletItemCarve(self,
                        stopWordPath = "./data_preprocess/chineseStopWords.txt",
                        loop=0,
                        carveDataPath = "./data_fasttext/",
                        carveDataName = "output.txt",
                        outlet = True,
                        batch = False):
        self._stopwordslist(stopWordPath)
        self._suggestFreq()
        num = 1
        items = self.data['ITEM_NAME']
        batch_count = 0
        if loop == 0:
            loop = len(items)
        for item in tqdm(items):
            seg = jieba.lcut(self.clean_numbers(item), cut_all=False)

            newSeg = []
            while (len(seg)):
                #print(len(seg))
                word = seg.pop(0)
                if word != '' and word != ' ':
                    if word not in self.stopWords:
                        if word != '\t' and word != '\r\n':
                            newSeg.append(word)
            tmp = " ".join(newSeg)
            if(tmp == ''):
                tmp = '无文本'
            self.seg_list.append(tmp)

            del newSeg

            if num >= loop:
                if(batch_count > 0):
                    self.outletCarveData(carveDataPath, carveDataName, batch = True)
                else:
                    self.outletCarveData(carveDataPath, carveDataName)
                return None

            if num % 10000 == 0 and outlet:
                self.outletCarveData(carveDataPath, carveDataName, batch = True)
                batch_count += 1
                
            num += 1
            
    def clean_numbers(self,s):
        #s = re.sub(r'[0-9]*[a-zA-Z]*[0-9]+[a-zA-Z]*'," ",s)
        s = re.sub(r'[0-9]+.[0-9]+'," ",s)
        s = re.sub(r'[0-9]+', " ", s)
        pattern = re.compile(r'[a-z]+', re.I)
        s = re.sub(pattern, lambda m: m.group(0) + " ",s)
        #s = re.sub('[a-z]{1,1}',"", s)
        return s

if __name__ == '__main__':
    DP = DataPreprocess()
    testData = DP.getInputData()
    DP.outletItemCarve(loop = 0)
