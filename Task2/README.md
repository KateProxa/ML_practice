# Практическое задание №2
Разработка собственного конвейера автоматизации для проекта машинного обучения
--
Для выполнения задания был выбран датасет о качестве вина с платформы kaggle [red-wine-quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) \
Решается задача классификации, целевая переменная - качество вина (оценка от 0 до 10)

## Этапы работы:
1. Локально развернут сервер с Jenkins, установлено необходимое программное обеспечение для работы над созданием модели машинного обучения.
2. Выбран способ получения данных - скачан датасет с платформы kaggle.
3. Проведена обработка данных, выделены важные признаки, сформированы датасеты для тренировки и тестирования модели.
4. Создана и обучена на тренировочном датасете модель машинного обучения и сохранена в pickle.
5. Проанализировано ее качество на тестовых данных.
6. Реализован конвейер

   
