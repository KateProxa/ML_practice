import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler  


def data_preparation(path):
    #Загрузка датафрейма
    df = pd.read_csv(path, encoding='utf-8', sep=',')  # импорт
    
    # удаление целевой переменной
    data = df.drop('quality', axis=1)  
    columns = data.columns

    # стандартизация данных
    standarted_data = StandardScaler()
    standarted_data.fit(data)

    # применение трансформера
    stdata = standarted_data.transform(data[columns])
    data_st = pd.DataFrame(stdata, columns=columns)

     # масштабирование данных
    scale_data = MinMaxScaler()
    scale_data.fit(data_st)
    
    # применение трансформера
    MM_data = scale_data.transform(data_st[columns])
    data_preprocessing = pd.DataFrame(MM_data, columns=columns)

    data_prep = pd.concat([data_preprocessing, df['quality']], axis=1)  
    return data_prep

