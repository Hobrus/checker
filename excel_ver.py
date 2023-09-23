import pandas as pd
import numpy as np
import requests
from sklearn.metrics import fbeta_score, precision_score, recall_score
from io import BytesIO
from time import time

# Загрузка dataset1, dataset2, dataset3 и dataset4
dataset1 = pd.read_excel("dataset1.xlsx")
dataset2 = pd.read_excel("dataset2.xlsx")
dataset3 = pd.read_excel("dataset3.xlsx")
dataset4 = pd.read_excel("dataset4.xlsx")

# datasets = [dataset1, dataset2, dataset3, dataset4]
# for ds in datasets:
#     if 'channel_id' in ds.columns:
#         ds.drop('channel_id', axis=1, inplace=True)

# URL вашего API
API_URL = "http://localhost:8000/api/zagruzka"

# Отправляем dataset1.xlsx на API и замеряем время
start_time = time()
output_buffer = BytesIO()
dataset1.to_excel(output_buffer, index=False, engine='openpyxl')
output_buffer.seek(0)

response = requests.post(API_URL, files={'file': ('dataset1.xlsx', output_buffer, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
if response.status_code == 200:
    API_URL = "http://localhost:8000/api/vigruzka_for_chek"
    response = requests.post(API_URL)

# Если ответ успешен, преобразуем его в DataFrame
if response.status_code == 200:
    predicted_data = pd.read_excel(BytesIO(response.content))

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

    # Второй вызов к API с dataset3
    output_buffer2 = BytesIO()
    dataset3.to_excel(output_buffer2, index=False, engine='openpyxl')
    output_buffer2.seek(0)

    API_URL = "http://localhost:8000/api/zagruzka"
    response_dataset3 = requests.post(API_URL, files={'file': ('dataset3.xlsx', output_buffer2, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
    if response_dataset3.status_code == 200:
        API_URL = "http://localhost:8000/api/vigruzka_for_chek"
        response_dataset3 = requests.post(API_URL)
    if response_dataset3.status_code == 200:
        predicted_data_from_dataset3 = pd.read_excel(BytesIO(response_dataset3.content))

        # Вместо преобразования всего датафрейма в список, вы преобразуете только столбец 'text'
        dataset3_texts = dataset3['text'].values.tolist()
        dataset4_texts = dataset4['text'].values.tolist()
        predicted_texts = predicted_data_from_dataset3['text'].values.tolist()

        # Теперь, когда вы проверяете, содержится ли текст в другом датафрейме, вы будете проверять только списки текстов
        y_true = [1 if entry in dataset4_texts else 0 for entry in dataset3_texts]
        y_pred = [1 if entry in predicted_texts else 0 for entry in dataset3_texts]
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f2_micro_score_new = fbeta_score(y_true, y_pred, beta=1, average='micro')
        print(f"New Logic - Precision: {precision}")
        print(f"New Logic - Recall: {recall}")
        print(f"New Logic - F2 micro score: {f2_micro_score_new}")
    else:
        print(f"Second API request with dataset3 failed with status code {response_dataset3.status_code}. Response text: {response_dataset3.text}")
    end_time = time()
    print(f"API execution time: {end_time - start_time} seconds")
else:
    print(f"API request failed with status code {response.status_code}. Response text: {response.text}")
