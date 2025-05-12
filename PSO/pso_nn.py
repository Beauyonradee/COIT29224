#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 20:57:37 2025

@author: yonradeelimsawat
"""

import random
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Suppress TensorFlow and other warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define PSO Parameters
SWARM_SIZE = 10
DIMENSIONS = 4  # learning rate, n1, n2, batch size
INFORMANTS = 3
NUM_GENERATIONS = 5
W = 0.729
C1 = 1.49
C2 = 1.49

# Define hyperparameter search boundaries
MIN_BOUNDARY = [0.0001, 4, 4, 8]   # Learning rate, n1, n2, batch
MAX_BOUNDARY = [0.01, 64, 64, 64]
desired_precision = 1e-5

# Define fitness function using Keras
def fitness_function(position):
    learning_rate = position[0]
    n1 = int(position[1])
    n2 = int(position[2])
    batch_size = int(position[3])

    model = Sequential()
    model.add(Dense(n1, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(n2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=30, batch_size=batch_size, callbacks=[early_stop])

    val_acc = max(history.history['val_accuracy'])
    return 1 - val_acc

# Particle Class
class Particle:
    def __init__(self):
        self.position = [
            random.uniform(MIN_BOUNDARY[0], MAX_BOUNDARY[0]),
            random.uniform(MIN_BOUNDARY[1], MAX_BOUNDARY[1]),
            random.uniform(MIN_BOUNDARY[2], MAX_BOUNDARY[2]),
            random.uniform(MIN_BOUNDARY[3], MAX_BOUNDARY[3])
        ]
        self.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
        self.fitness = fitness_function(self.position)
        self.best_position = list(self.position)
        self.best_fitness = self.fitness
        self.informants = random.sample(range(SWARM_SIZE), INFORMANTS)
        self.group_best_position = list(self.position)
        self.group_best_fitness = self.fitness

    def update_velocity(self):
        for d in range(DIMENSIONS):
            r1, r2 = random.random(), random.random()
            cognitive = C1 * r1 * (self.best_position[d] - self.position[d])
            social = C2 * r2 * (self.group_best_position[d] - self.position[d])
            self.velocity[d] = W * self.velocity[d] + cognitive + social

    def update_position(self):
        for d in range(DIMENSIONS):
            self.position[d] += self.velocity[d]
            self.position[d] = max(MIN_BOUNDARY[d], min(MAX_BOUNDARY[d], self.position[d]))
        self.fitness = fitness_function(self.position)

    def update_group_best(self, swarm):
        best_informant = min(self.informants, key=lambda i: swarm[i].best_fitness)
        if swarm[best_informant].best_fitness < self.group_best_fitness:
            self.group_best_fitness = swarm[best_informant].best_fitness
            self.group_best_position = list(swarm[best_informant].best_position)

# Initialize swarm
swarm = [Particle() for _ in range(SWARM_SIZE)]
global_best = min(swarm, key=lambda p: p.best_fitness)
global_best_position = list(global_best.best_position)
global_best_fitness = global_best.best_fitness

# PSO main loop
fitness_progress = []

for gen in range(NUM_GENERATIONS):
    for particle in swarm:
        particle.update_group_best(swarm)
        particle.update_velocity()
        particle.update_position()

        if particle.fitness < particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_position = list(particle.position)

    best_particle = min(swarm, key=lambda p: p.best_fitness)
    if best_particle.best_fitness < global_best_fitness:
        global_best_fitness = best_particle.best_fitness
        global_best_position = list(best_particle.best_position)

    fitness_progress.append(1 - global_best_fitness)  # Store accuracy

    print(f"Generation {gen + 1}: Best Validation Accuracy = {1 - global_best_fitness:.4f}")

    if global_best_fitness < desired_precision:
        print("Desired precision reached.")
        break

print("\nOptimization Complete!")
print(f"Best Learning Rate: {global_best_position[0]:.6f}")
print(f"Neurons Layer 1: {int(global_best_position[1])}")
print(f"Neurons Layer 2: {int(global_best_position[2])}")
print(f"Batch Size: {int(global_best_position[3])}")
print(f"Best Validation Accuracy: {1 - global_best_fitness:.4f}")

# Plot PSO fitness progression
plt.plot(fitness_progress, label='Best Fitness over Generations')
plt.title('PSO Fitness Progression')
plt.xlabel('Generations')
plt.ylabel('Fitness (1 - Accuracy)')
plt.legend()
plt.show()


