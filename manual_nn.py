#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 20:01:37 2025

@author: yonradeelimsawat
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart.csv")
print(df.head())
print(df.info())

# drop rows with missing data
df.dropna(inplace=True)

#Data Proprocessing
x = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

"Build Neural Network"
def build_nn(learning_rate=0.001):
    model = Sequential()
    model.add(Dense(16, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

model = build_nn()
history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred = (model.predict(x_test) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test, y_pred))




