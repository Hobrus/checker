import pandas as pd
import numpy as np
import requests
from sklearn.metrics import fbeta_score, precision_score, recall_score
from io import StringIO
from time import time

# Загрузка dataset1 и dataset2
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")

# URL вашего API
API_URL = "https://your_api_endpoint.com"

# Отправляем dataset1.csv на API и замеряем время
start_time = time()
response = requests.post(API_URL, files={'file': StringIO(dataset1.to_csv(index=False))})
end_time = time()

# Если ответ успешен, преобразуем его в DataFrame
if response.status_code == 200:
    predicted_data = pd.read_csv(StringIO(response.text))

    # Заполняем NaN значения в столбце категории случайными категориями
    predicted_data['category'].fillna(np.random.choice(dataset2['category'].unique()), inplace=True)

    # Старая логика
    common_texts = set(dataset1['text']).intersection(set(predicted_data['text']))
    dataset2_filtered = dataset2[dataset2['text'].isin(common_texts)]
    predicted_data_filtered = predicted_data[predicted_data['text'].isin(common_texts)]
    merged_data = dataset2_filtered.merge(predicted_data_filtered, on='text', how='inner', suffixes=('_true', '_pred'))
    y_true_category = merged_data['category_true']
    y_pred_category = merged_data['category_pred']
    f2_micro_score_old = fbeta_score(y_true_category, y_pred_category, beta=2, average='micro')

    print(f"Old Logic - F2 micro score: {f2_micro_score_old}")

    # Новая логика
    y_true = dataset1['text'].isin(dataset2['text']).astype(int)
    y_pred = (~dataset1['text'].isin(predicted_data['text'])).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f2_micro_score_new = fbeta_score(y_true, y_pred, beta=2)

    print(f"New Logic - Precision: {precision}")
    print(f"New Logic - Recall: {recall}")
    print(f"New Logic - F2 micro score: {f2_micro_score_new}")

    print(f"API execution time: {end_time - start_time} seconds")
else:
    print(f"API request failed with status code {response.status_code}. Response text: {response.text}")