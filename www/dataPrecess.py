import json
from www import FC
from www import DP
import os
import uuid
import pandas as pd

def dataPrecess(str, data):
    id = str(uuid.uuid3(uuid.NAMESPACE_DNS, 'data'))
    filename = id + ".json"
    basepath = os.path.dirname(__file__)
    if str == "text":
        DP.data = data

        temp_path = os.path.join(basepath, 'static\\temp')
        DP.outletItemCarve(carveDataPath=temp_path,
                           carveDataName=id + "Carve.txt")

        carveData = DP.outletTextCarve()

        pred = FC.predict_inputText(carveData)

        jsondata = {"data": data, "type": pred}

        save_path = os.path.join(basepath, 'static\\json')
        file_path = os.path.join(save_path, filename)
        with open(file_path, "w") as file:
            json.dump(jsondata, file)
        return id

    elif str == "file":
        input_path = os.path.join(basepath, 'static\\uploads')
        inputText = DP.getInputData(path=input_path, name=data)

        temp_path = os.path.join(basepath, 'static\\temp')
        DP.outletItemCarve(carveDataPath=temp_path,
                           carveDataName=id + "Carve.txt")

        pred_path = os.path.join(basepath, 'static\\temp')
        FC.predict_inputFile(inputpath=temp_path,
                             name=id + "Carve.txt",
                             outputpath=pred_path + id + ".txt")
        pred = pd.read_table(pred_path + id + ".txt",
                             encoding='utf-8',
                             header=None,
                             quoting=3,)

        res = pd.concat([inputText, pred], axis=1)
        res.columns = ['text', 'pred']
        jsondata = {"data": [{"item": list(res.loc[i, :])[0], "type": list(res.loc[i, :])[1]} for i in range(len(res))]}

        save_path = os.path.join(basepath, 'static\\json')
        file_path = os.path.join(save_path, filename)
        with open(file_path, "w") as file:
            json.dump(jsondata, file)
        return id
    else:
        return "0"