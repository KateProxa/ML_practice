# Практическое задание №1
Создание автоматического конвейера проекта машинного обучения с применением простых скриптов автоматизации

## Этапы работы:
1. Создан python-скрипт [data_creation.py](https://github.com/KateProxa/ML_practice/blob/main/Task1/data_creation.py),
   который создает набор данных, а также разбивает на тренировочную и тестовую части
2. Создан python-скрипт [model_preprocessing.py](https://github.com/KateProxa/ML_practice/blob/main/Task1/model_preprocessing.py), который выполняет предобработку данных
3. Создан python-скрипт [model_preparation.py](https://github.com/KateProxa/ML_practice/blob/main/Task1/model_preparation.py), который создает и обучает модель
   машинного обучения на построенных данных методом опорных векторов из папки [train](https://github.com/KateProxa/ML_practice/tree/main/Task1/train). Метрика - MSE
4. Создан python-скрипт [model_testing.py](https://github.com/KateProxa/ML_practice/blob/main/Task1/model_testing.py), проверяющий модель машинного обучения
   на построенных данных из папки [«test»](https://github.com/KateProxa/ML_practice/tree/main/Task1/test)
5. Написан bash-скрипт [pipeline.sh](https://github.com/KateProxa/ML_practice/blob/main/Task1/pipeline.sh) последовательно запускающий все python-скрипты
