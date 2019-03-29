import pandas as pd
import numpy as np
import jieba
from gensim.models import word2vec
from gensim import corpora
import os
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class TrainDataPrecondition(object):
    def __init__(self):
        self.seg_list = []
        self.stopWords = set()
        self.trainData = None
        self.carveData = []
        self.model = None
        self.wv = None
        self.corpus = None
        self.w2vDim = 100
        self.EmbeddingMatrix = []
        self.classNum = None
        self.vocbSize = None
        self.maxWordNum = 0
        self.testData = None
        

    def getTrainData(self, path='./data_LC/', name="train.tsv"):
        print("Reading TrainData")
        trainData = pd.read_table(path + name,
                                  encoding="gb18030",
                                  quoting=3
                                  )
        self.trainData = trainData
        trainType = trainData['TYPE']
        return trainData['ITEM_NAME']
    
    def getTestData(self, path='./data_LC/', name="test.tsv"):
        print("Reading TestData")
        testData = pd.read_table(path + name,
                                  encoding="gb18030",
                                  quoting=3
                                  )
        self.testData = testData
        return testData
    
    def _stopwordslist(self, stopwordpath):
        print("insert stopwords")
        stopwords = [line.strip() for line in open(stopwordpath, 'r', encoding='gb18030').readlines()]
        self.stopWords = set(stopwords)

    def _suggestFreq(self):
        freq = ['腾讯QQ', 'q币', 'qq币', 'Q币', 'QQ币', 'QB', 'qb', 'DVD','黄钻','红钻','黑钻','绿钻','狗粮','成犬'
               ,'狗盆','猫碗','狗碗','泡茶','煮水','煮茶','蓝钻','紫钻']
        for word in freq:
            jieba.suggest_freq(word, True)

    def outletCarveData(self, path, name, num):
        if not os.path.exists(path):
            os.mkdir(path)
        saveData = pd.DataFrame(self.seg_list)
        saveData.to_csv(path + str(num) + name,
                        encoding="gb18030",
                        header=False,
                        index=False)
        del saveData
        del self.seg_list
        self.seg_list = []
        #将pandas数据存成文件

    def outletItemCarve(self,
                        stopWordPath="./data_preprocess/chineseStopWords.txt",
                        loop=0,
                        carveDataPath="./data_preprocess/carve/",
                        carveDataName="carve.tsv",
                        outlet=True,
                        batch=False):
        self._stopwordslist(stopWordPath)
        self._suggestFreq()
        num = 1
        items = tqdm(pd.DataFrame(self.trainData, columns=['ITEM_NAME'])['ITEM_NAME'])
        if loop == 0:
            loop = len(items)
        for item in items:
            seg = jieba.lcut(self.clean_numbers(item), cut_all=False)

            newSeg = []
            while (len(seg)):
                #print(len(seg))
                word = seg.pop(0)
                if word != '' and word != ' ':
                    if word not in self.stopWords:
                        if word != '\t' and word != '\r\n':
                            newSeg.append(word)
            self.seg_list.append(newSeg)

            del newSeg

            if num >= loop:
                self.outletCarveData(carveDataPath, carveDataName, num)
                return None

            if batch:
                if num % 10000 == 0 and outlet:
                    self.outletCarveData(carveDataPath, carveDataName, num)

            num += 1


    def getCarveData(self, path="./data_preprocess/carve/", addNone=False):
        print("getCarveData")
        for dirpath, dirnames, filenames in os.walk(path):
            for filepath in filenames:
                getCarveData = pd.read_csv(dirpath+filepath,
                                           encoding="gb18030",
                                           header=None,
                                           low_memory=False
                                           )
                getCarveData = getCarveData.values
                #print("getData:", getCarveData[0])

                for words in getCarveData:
                    tempWords = []
                    for word in words:
                        if word != None:
                            if isinstance(word, str):
                                tempWords.append(word)
                            elif addNone:
                                tempWords.append('')
                    self.carveData.append(tempWords)
                    #print(len(tempWords))
                    #print(tempWords)
                    if len(tempWords) > self.maxWordNum:
                        self.maxWordNum = len(tempWords)
                    #print(self.maxWordNum)
                    del tempWords
                #print("carveData:", self.carveData[0])
                del getCarveData
        return self.carveData

    def outletTypeData(self, path="./data_preprocess/type/", name="type.tsv"):
        if not os.path.exists(path):
            os.mkdir(path)
        typeData = pd.DataFrame(self.trainData, columns=['TYPE'])
        # 将pandas数据存成文件
        typeData.to_csv(path + name,
                        encoding="gb18030",
                        header=False,
                        index=False)

    def getTypeData(self, path="./data_preprocess/type/", name="type.tsv"):
        print("getTypeData")
        typeData = pd.read_csv(path + name,
                               encoding="gb18030",
                               header=None)
        typeData = typeData.values

        typeList = [type[0] for type in typeData]
        typeSet = set(typeList)

        type2typeidx = {types: idx for idx, types in enumerate(typeSet)}
        typeidx2type = {idx: types for (types, idx) in type2typeidx.items()}
        #idx2typeidx = {idx: type2typeidx[types] for idx, types in enumerate(typeList)}
        typeidxList = [type2typeidx[types] for types in typeList]
        self.classNum = len(typeidxList)
        with open('./data_preprocess/type.txt','w+') as f:
            for idx, tp in typeidx2type.items():
                f.write(str(idx)+ " "+ tp + "\n")
        return typeidx2type, typeidxList

    def w2vTraining(self, loop=5, path="./data_preprocess/model/", name="model.model"):
        flag = True
        times = 0
        while times < loop:
            times += 1
            if flag:
                flag = False
                if os.path.exists(path + name):
                    self.getModel(path, name)
                else:
                    print("w2vTraining")
                    #print("training NSmodelCBOW")
                    print("创建模型存于%s%s" % (path, name))
                    print("训练模型第%d次" % times)
                    self.model = word2vec.Word2Vec(self.carveData, workers=4,
                                                    hs=0, min_count=3,
                                                    window=6, size=self.w2vDim,
                                                    sg=0, iter=1)
                    # print("training NSmodelSG")
                    # NSmodelSG = word2vec.Word2Vec(self.carveData, hs=0, min_count=2, window=6, size=100, sg=1, iter=8)
                    # print("training HSmodelCBOW")
                    # HSmodelCBOW = word2vec.Word2Vec(self.carveData, hs=1, min_count=2, window=6, size=100, sg=0, iter=10)
                    # print("training HSmodelSG")
                    # HSmodelSG = word2vec.Word2Vec(self.carveData, hs=1, min_count=2, window=6, size=100, sg=1, iter=8)
                    self._saveModel(self.model, path, name)
                    continue
            print("训练模型第%d次" % times)
            self.model.build_vocab(self.carveData, update=True)
            self.model.train(self.carveData,
                             total_examples=self.model.corpus_count,
                             epochs=1)
            self._saveModel(self.model, path, name)


    def trainingTest(self, model=None):
        if model is None:
            model = self.model
        def nearestWord(searchWord, count=5):
            req_count = count
            for key in model.wv.similar_by_word(searchWord, topn=100):
                if len(key[0]) == 3:
                    req_count -= 1
                    print(key[0], key[1])
                    if req_count == 0:
                        break

        def wordSimilaity(string1, string2):
            #cbow,
            ns_cbow = model.wv.similarity(string1, string2)
            #self.savemodel(ns_cbow, "./data_preprocess/model/ns_cbow.model")

            #ns_sg = NSmodelSG.wv.similarity(string1, string2)
            #self.savemodel(ns_sg, "./data_preprocess/model/ns_sg.model")

            #hs_cbow = HSmodelCBOW.wv.similarity(string1, string2)
            #self.savemodel(hs_cbow, "./data_preprocess/model/hs_cbow.model")

            #hs_sg = HSmodelSG.wv.similarity(string1, string2)
            #self.savemodel(hs_sg, "./data_preprocess/model/hs_sg.model")

            print("ns_cbow:", ns_cbow)
            #print("ns_sg:", ns_sg)
            #print("hs_cbow:", hs_cbow)
            #print("hs_sg:", hs_sg)

        searchWord = '爱普生'
        print('NSmodelCBOW')
        nearestWord(searchWord)
        #print('\nNSmodelSG')
        #nearestWord(NSmodelSG, searchWord)
        #print('\nHSmodelCBOW')
        #nearestWord(HSmodelCBOW, searchWord)
        #print('\nHSmodelSG')
        #nearestWord(HSmodelSG, searchWord)

        wordSimilaity("爱普生", "EPSON")

    def _saveModel(self, model, path="./data_preprocess/model/", name="model.model"):
        if not os.path.exists(path):
            os.mkdir(path)
        model.save(path+name)

    def getModel(self, path="./data_preprocess/model/", name="model.model"):
        if os.path.exists(path+name):
            self.model = word2vec.Word2Vec.load(path+name)

    #弃用
    def getCorpora(self):
        print("getCorpora")
        dictionary = corpora.Dictionary(self.carveData)
        corpus_idx = [dictionary.doc2idx(text) for text in self.carveData]
        corpus_bow = [dictionary.doc2bow(text) for text in self.carveData]

        return corpus_idx

    def saveModelAsWordVector(self, path="./data_preprocess/wordVec/", name="wordVec.wv"):
        #模型训练好以后可以转成WordVector模式，体积小速度快
        print("saveModelAsWordVector")
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.wv.save(path + name)

    def getWordVector(self, path="./data_preprocess/wordVec/", name="wordVec.wv"):
        if os.path.exists(path+name):
            self.wv = KeyedVectors.load(path+name, mmap='r')  # 内存映射
        return self.wv

    def getVecList(self):
        #使用wordVector，需要先将model转成wv
        print("getVecList")
        vecList = []
        for words in self.carveData:
            tempList = []
            for word in words:
                try:
                    tempList.append(self.wv[word])
                except KeyError:
                    pass#训练时被排除的少用词
            vecList.append(tempList)
            del tempList
        return vecList

    def getDictionary(self, carveData=None):
        if carveData == None:
            carveData = self.carveData
        wordSet = set([word for words in carveData for word in words])
        word2wordidx = {word: idx for idx, word in enumerate(wordSet)}
        wordidx2word = {idx: word for (word, idx) in word2wordidx.items()}
        return word2wordidx, wordidx2word

    def getCorpus(self, word2wordidx,
                  addNone=False, matrix=False,
                  carveData=None, wv=None,
                  maxWordNum=None):

        if carveData==None:
            carveData = self.carveData
        if wv == None:
            wv = self.wv

        if matrix:

            if maxWordNum == None:
                maxWordNum = self.maxWordNum

            self.corpus = np.zeros([len(carveData), maxWordNum])
            i = 0
            for words in carveData:
                j = 0
                for word in words:
                    if word in self.wv:
                        self.corpus[i][j] = word2wordidx[word]
                    elif addNone:
                        self.corpus[i][j] = word2wordidx['']
                    j += 1
                for last in range(j, maxWordNum):
                    self.corpus[i][last] = word2wordidx['']
                i += 1

        else:
            self.corpus = []
            for words in self.carveData:
                tempList = []
                for word in words:
                    if word in self.wv:
                        tempList.append(word2wordidx[word])
                self.corpus.append(tempList)
                del tempList

        return self.corpus


    def getEmbeddingMatrix(self, wordidx2word):
        self.vocbSize = len(wordidx2word)
        for index in range(self.vocbSize):
            if wordidx2word[index] in self.wv:
                self.EmbeddingMatrix.append(self.wv[wordidx2word[index]])
            else:
                self.EmbeddingMatrix.append([0.0 for i in range(self.w2vDim)])

        return np.array(self.EmbeddingMatrix)
    
    def get_fasttextData(self,data = None, labels = None, path = "./data_fasttext/", name = 'train.txt'):
        d_list = []
        data = tqdm(data)
        for i, item in enumerate(data):
            outline = " ".join(item) + "\t__label__" + str(labels[i])
            d_list.append(outline)
        df = pd.DataFrame(d_list)
        df.to_csv(path + name,index=False, encoding = 'gb18030', header = None)
        
    def clean_numbers(self,s):
        #s = re.sub(r'[0-9]*[a-zA-Z]*[0-9]+[a-zA-Z]*'," ",s)
        s = re.sub(r'[0-9]+.[0-9]+'," ",s)
        s = re.sub(r'[0-9]+', " ", s)
        pattern = re.compile(r'[a-z]+', re.I)
        s = re.sub(pattern, lambda m: m.group(0) + " ",s)
        #s = re.sub('[a-z]{1,1}',"", s)
        return s

if __name__ == "__main__":
    TDP = TrainDataPrecondition()
    TDP.getTrainData()
    X_train = TDP.getCarveData()

    #X_test = TDP.getTestData()
    #TDP.outletItemCarve(loop=0)
    #TDP.outletTypeData()
    type_dict, type_list = TDP.getTypeData()
    #carveData = TDP.getCarveData("./data_preprocess/carve/", addNone=True)
    y_train = type_list
    X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=233)
    TDP.get_fasttextData(X_tra, y_tra, name = 'train.txt')
    TDP.get_fasttextData(X_val, y_val, name = 'test.txt')
    #如果没训练就需要getModel，训练了就不需要了，可以调用trainingTest测试模型效果
    #可以增量训练
    #TDP.w2vTraining(loop=1, path="./data_preprocess/model/", name="NSmodelCBOW_mc3w6.model")
    #TDP.getModel(path="./data_preprocess/model/", name="NSmodelCBOW_mc3w6.model")
    #TDP.trainingTest()

    # 模型训练好以后可以转成WordVector模式，体积小速度快,转换也需要getWordVector()
    #TDP.saveModelAsWordVector(path="./data_preprocess/wordVec/", name="wordVec.wv")
    #wv = TDP.getWordVector(path="./data_preprocess/wordVec/", name="wordVec.wv")
    #将分词用词向量代替
    #vecList = TDP.getVecList()

    #w2i, i2w = TDP.getDictionary()
    #corpus = TDP.getCorpus(w2i, matrix=True)
    #EmbeddingMatrix = TDP.getEmbeddingMatrix(i2w)
