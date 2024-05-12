import pickle 
from model_preprocessing import data_preparation  

model_path = 'model.pkl'
test_path = 'test/test_data.csv'

# загрузка модели
loaded_model = pickle.load(open(model_path, 'rb'))  

# загрузка тестовыx данныx
df = data_preparation(test_path)  
X, y = df.drop(columns=['target']), df['target']
test_predict = loaded_model.predict(X)

print('Предсказания модели :', test_predict)