import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from time import time

# Загрузка dataset1, dataset2, dataset3 и dataset4
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")
dataset3 = pd.read_csv("dataset3.csv")
dataset4 = pd.read_csv("dataset4.csv")

test1 = pd.read_csv("test1.csv")
test2 = pd.read_csv("test2.csv")

# Здесь предположим, что test1 это "предсказанные" данные (аналогично ответу API) для dataset1, а test2 это "истинные" данные.
predicted_data = test1.copy()

# Заполняем NaN значения в столбце категории случайными категориями
predicted_data['category'].fillna(np.random.choice(dataset2['category'].unique()), inplace=True)
predicted_data['category'] = predicted_data['category'].str.lower()

# Старая логика
common_texts = set(dataset1['text']).intersection(set(predicted_data['text']))
dataset2_filtered = dataset2[dataset2['text'].isin(common_texts)]
predicted_data_filtered = predicted_data[predicted_data['text'].isin(common_texts)]
merged_data = dataset2_filtered.merge(predicted_data_filtered, on='text', how='inner', suffixes=('_true', '_pred'))
y_true_category = merged_data['category_true']
y_pred_category = merged_data['category_pred']
f2_micro_score_old = fbeta_score(y_true_category, y_pred_category, beta=1, average='micro')
print(f"Old Logic - F2 micro score: {f2_micro_score_old}")

# Вместо преобразования всего датафрейма в список, вы преобразуете только столбец 'text'
dataset3_texts = dataset3['text'].values.tolist()
dataset4_texts = dataset4['text'].values.tolist()
predicted_texts = test2['text'].values.tolist()

# Теперь, когда вы проверяете, содержится ли текст в другом датафрейме, вы будете проверять только списки текстов
y_true = [1 if entry in dataset4_texts else 0 for entry in dataset3_texts]
y_pred = [1 if entry in predicted_texts else 0 for entry in dataset3_texts]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f2_micro_score_new = fbeta_score(y_true, y_pred, beta=1, average='micro')
print(f"New Logic - Precision: {precision}")
print(f"New Logic - Recall: {recall}")
print(f"New Logic - F2 micro score: {f2_micro_score_new}")
