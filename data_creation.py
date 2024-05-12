import pandas as pd
import numpy as np 
from pathlib import Path 

np.random.seed(5)

# генерация данных
size_var = 10000
var_ = {
    'var1': np.random.poisson(10, size_var),
    'var2': np.random.rand(size_var),
    'var3': np.random.rand(size_var),
    'var4': np.random.rand(size_var),
    'var5': np.random.poisson(10, size_var)
}

# Добавляю целевую переменную
var_['target'] = (
        0.5 * var_['var1']
        + 0.8 * var_['var2']
        + 2.1 * var_['var3']
        + 0.3 * var_['var5']
        + np.random.exponential(scale=0.5, size=size_var)
)

# Coздание DataFrame
df_train = pd.DataFrame(var_).sample(frac=0.9, random_state=5)
df_test = pd.DataFrame(var_).drop(df_train.index)

# Coхранение DataFrame
path_train = Path('train/train_data.csv')
path_train.parent.mkdir(parents=True, exist_ok=True)

path_test = Path('test/test_data.csv')
path_test.parent.mkdir(parents=True, exist_ok=True)

df_train.to_csv(path_train, index=False)
df_test.to_csv(path_test, index=False)