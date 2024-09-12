import dill
import json
import pandas as pd
import os
from datetime import datetime


path = os.path.dirname(os.path.abspath('.'))

def predict():
    with open(f'{path}/data/models/cars_pipe_202409121448.dill', 'rb') as f:
        df2 = pd.DataFrame({'predictions': ""}, index=[7310993818, 7313922964, 7315173150, 7316152972, 7316509996])
        load = dill.load(f)
        model = load['model']
        with open(f'{path}/data/test/7310993818.json', 'rb') as f1, open(f'{path}/data/test/7313922964.json', 'rb') as f2, open(f'{path}/data/test/7315173150.json', 'rb') as f3, open(f'{path}/data/test/7316152972.json', 'rb') as f4, open(f'{path}/data/test/7316509996.json', 'rb') as f5:
            list1 = [f1, f2, f3, f4, f5]
            for i in range(len(list1)):
                form = json.load(list1[i])
                df = pd.DataFrame([form])
                open_file = f'{path}/data/test/{list1[i]}.json'
                file_name = os.path.basename(open_file)
                name = file_name.split('.')[0]
                print(f"{name}: {model.predict(df)[0]}")
                df2.iloc[i] = model.predict(df)[0]
            df2.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', encoding="utf-8")




if __name__ == '__main__':
    predict()
