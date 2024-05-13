import pandas as pd
from sklearn.model_selection import train_test_split
import opendatasets as od

# загружаем датасет
od.download('https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/download?datasetVersionNumber=2', data_dir='./Task2/')
df = pd.read_csv('./Task2/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

# разбиваем на тренировочную и валидационную выборки
X_train, X_val, = train_test_split(df, test_size=0.7, random_state=5)

# сохранение данных в csv
X_train.to_csv('./Task2/data_train.csv', index=False)
X_val.to_csv('./Task2/data_test.csv', index=False)

print('Tренировочные данные: сохранены')
print('Тестовые данные: сохранены')