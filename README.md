# Abnormal-Activity-Detection

## Overview

This script performs object tracking using YOLOv8 and DeepSORT, followed by activity classification using LSTM and Time Series Classifier models. The process includes setting up the environment, running object detection, and classifying activities based on detected objects.

## Prerequisites

- **Google Colab**: Ensure you are using GPU runtime.
- **Dependencies**: Required libraries will be installed automatically.

## Setup

1. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Clone the GitHub Repository**:
   ```python
   !git clone https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking.git
   ```

3. **Navigate to Project Directory**:
   ```python
   %cd /content/drive/MyDrive/YOLOv8-DeepSORT-Object-Tracking
   ```

4. **Install Dependencies**:
   ```python
   !pip install -e '.[dev]'
   ```

5. **Navigate to YOLO Detection Directory**:
   ```python
   %cd /content/drive/MyDrive/YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect
   ```

6. **Download and Unzip DeepSORT Files**:
   ```python
   !gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"
   !unzip 'deep_sort_pytorch.zip'
   ```

## Running the Script

1. **Run Object Detection**:
   ```python
   import os
   avi_directory = "/content/drive/MyDrive/YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/"
   avi_files = [file for file in os.listdir(avi_directory) if file.endswith(".avi") or file.endswith(".mp4")]
   numbered_avi_files = [f"{file}_{i+1}" for i, file in enumerate(avi_files)]
   source_param = '"' + ', '.join(numbered_avi_files) + '"'
   print(source_param)
   !python predict.py --multirun model=yolov8l.pt source={source_param}
   ```

2. **LSTM Classifier**:
   ```python
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import LabelEncoder
   from sklearn.metrics import precision_recall_fscore_support
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense

   # Load and preprocess data
   data = pd.read_csv("output_1.csv")
   label_encoder = LabelEncoder()
   data['Class '] = label_encoder.fit_transform(data['Class '])
   sequence_length = 10
   sequences = []
   labels = []

   for i in range(len(data) - sequence_length + 1):
       sequence = data.iloc[i:i+sequence_length]
       sequences.append(sequence[['Video Number', 'Frame Number', 'Person Number', 'Left Coordinate',
          'Top Coordinate', 'Width', 'Height']].values)
       labels.append(sequence['Class '].values[-1])

   sequences = np.array(sequences)
   labels = np.array(labels)

   X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.4, random_state=42)
   model = Sequential()
   model.add(LSTM(64, input_shape=(sequence_length, 7)))
   model.add(Dense(1, activation='sigmoid'))
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
   ```

3. **Time Series Classifier**:
   ```python
   pip install tsai

   from tsai.all import *
   import sklearn.metrics as skm

   # Load data and preprocess
   data = pd.read_csv('output_1.csv')
   X = data.drop(columns=['Class '])
   y = data['Class ']
   label_encoder = LabelEncoder()
   y = label_encoder.fit_transform(y)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
   X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
   X = X.astype(np.float32)
   y = y.astype(np.float32)
   tfms = [None, TSClassification()]
   batch_tfms = [TSStandardize(by_sample=True)]
   learn = TSClassifier(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy, arch=InceptionTimePlus, arch_config=dict(fc_dropout=.5), train_metrics=True)
   learn.fit_one_cycle(10, 1e-2)
   learn.export("mv_clf.pkl")
   learn.plot_metrics()
   learn.plot_confusion_matrix()
   ```

## Results

- **LSTM Classifier**: Outputs loss, accuracy, precision, recall, and F1 score.
Confusion Matrix:
<img width="521" alt="Screenshot 2024-07-25 at 23 39 31" src="https://github.com/user-attachments/assets/25c07388-bf11-4417-93ad-ab0a78f76d6d">
Outputs loss, accuracy, precision, recall, and F1 score: 
<img width="748" alt="Screenshot 2024-07-25 at 23 40 56" src="https://github.com/user-attachments/assets/5a4fa161-2f9c-44e5-8b53-74e5f36faa05">

<img width="669" alt="Screenshot 2024-07-25 at 23 39 45" src="https://github.com/user-attachments/assets/b7643c62-b879-4560-8cc2-5bd678f24e76">

- **Time Series Classifier**: Outputs accuracy, precision, recall, and F1 score, and generates plots for metrics and confusion matrix.
Confusion Matrix:
<img width="324" alt="Screenshot 2024-07-25 at 23 40 02" src="https://github.com/user-attachments/assets/278e2d26-900c-479e-9b66-736bc43da679">

Outputs loss, accuracy, precision, recall, and F1 score: 
<img width="174" alt="Screenshot 2024-07-25 at 23 41 15" src="https://github.com/user-attachments/assets/6632f7dd-c598-4f5a-b017-6bf3ba68e730">


<img width="1070" alt="Screenshot 2024-07-25 at 23 40 30" src="https://github.com/user-attachments/assets/101a8c91-8111-4ff7-8b9d-741f20fc3f6b">
