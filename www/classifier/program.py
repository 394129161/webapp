from www.classifier.fasttextClassifier import FastTextClassifier
from www.classifier.inputDataProcess import DataPreprocess

def initClassifier():
    FC = FastTextClassifier()
    FC.load_model()
    FC.get_idx2type()
    return FC

def initInputDataProcess():
    DP = DataPreprocess()
    return DP
