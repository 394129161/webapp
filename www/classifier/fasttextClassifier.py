import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext
import pandas as pd
#from www.classifier.trainDataProcess import TrainDataPrecondition
#import os

class FastTextClassifier(object):
               
    def __init__(self):
        self.classifier = None
        self.result = None
        self.items = None
        self.idx2type = None
        
    def train_model(self, filepath = "./data_fasttext/train.txt", modelpath = "./model/LcClassifier.model"):
        print('start training model...')
        self.classifier = fasttext.supervised(filepath, 
                                     modelpath, 
                                     label_prefix="__label__", 
                                      encoding = "gb18030", 
                                     lr_update_rate = 1, 
                                     loss = 'hs', 
                                     word_ngrams = 2, 
                                     epoch = 30, 
                                     bucket = 2000000)
        return self.classifier
                   
    def load_model(self, path = "./model/", name = "LcClassifier.model.bin"):
        self.classifier = fasttext.load_model( path + name,
                                      label_prefix = '__label__',
                                      encoding = 'gb18030'
                                    )
        return self.classifier
               
    def run_testFile(self, path = "./data_fasttext/", name = 'test.txt'):
        print('run val data...')
        if(self.classifier == None):
            print("You haven't got a available classifier!")
            return None
        self.result = self.classifier.test(path + name)
        print("the result's precision is {:.2%}".format(self.result.precision))
        print(self.result.recall)
    
    def predict_inputText(self, text = None):
        if(self.classifier == None):
            print("You haven't got a available classifier!")
            return None
        if(text == ""):
            return "无法分类"
        pred_idx = self.classifier.predict([text])
        
        return self.idx2type[int(pred_idx[0][0])]
          
    def batchOutput(self, text,outputpath = './output/result.txt',batch = False):
        pred_list = self.classifier.predict(text, k = 1)
        type_list = []
                         
        for pred in pred_list:
            idx = pred[0]
            type_list.append(self.idx2type[int(idx)])
        type_df = pd.DataFrame(type_list)
        if(batch):
            type_df.to_csv(outputpath, index = False, encoding = 'utf-8', mode = 'a',header = None)
        else:
            type_df.to_csv(outputpath, index = False, encoding = 'utf-8', header = None)

    def predict_inputFile(self, inputpath = './data_fasttext/', name = 'output.txt', outputpath = './output/result.txt', batch = False):
        df = pd.read_table(inputpath + name, encoding = 'gb18030', header = None, quoting = 3, names = ['item'])
        texts = []
        num = 1
        length = df.shape[0]
        batch_count = 0
        for line in df['item']:
            line = line.strip()
            texts.append(line)

            if(num >= length):
                if(batch_count > 0):
                    self.batchOutput(texts, outputpath, batch = True)
                else:
                    self.batchOutput(texts, outputpath)
                texts = []
            if(num % 50000 == 0):
                batch_count += 1
                self.batchOutput(texts, outputpath, batch = True)
                texts = []
            num += 1
        
    def get_idx2type(self, path = './data_preprocess/', name = 'type.txt'):
        self.idx2type = {}
        with open(path + name, 'r') as f:
            for line in f.readlines():
                tmp = line.strip().split()
                idx = int(tmp[0])
                self.idx2type[idx] = tmp[1]
        return self.idx2type

if __name__ == "__main__":
    FC = FastTextClassifier()
    classifier = FC.train_model()
    #FC.run_testFile()
    idx2type = FC.get_idx2type()
    #pred = FC.predict_inputText("腾讯QQ 蓝 钻 贵族 直充")
    #print("the predict type is ", pred)
    #classifier = FC.load_model()
    FC.predict_inputFile()
