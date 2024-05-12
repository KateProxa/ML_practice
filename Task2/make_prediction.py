import pickle
import pandas as pd
from model_preprocessing import data_preparation

loaded_model = pickle.load(open('Task2/model.pkl', 'rb'))

# прочитаем из csv-файла подготовленный датасет для обучения
data_test = data_preparation('Task2/data_test.csv')

X, y = data_test.drop(columns=['quality']).values, data_test['quality'].values

# сделаем предсказание для первого вина из тестовой выборки
test_predict = loaded_model.predict(X[0:1])
print(test_predict.item())