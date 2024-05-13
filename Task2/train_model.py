
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from model_preprocessing import data_preparation

# прочитаем из csv-файла подготовленный датасет для обучения
data_train = data_preparation('./Task2/data_train.csv')

# Убираем целевую переменную
X, y = data_train.drop(columns=['quality']).values, data_train['quality'].values

#Разбиваем на тестовую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.9,
                                                  random_state=5)

model = LogisticRegression(max_iter=100_000).fit(X_train, y_train)

# сохраним обученную модель
pickle.dump(model, open('./Task2/model.pkl', 'wb'))

print('Модель сохранена')