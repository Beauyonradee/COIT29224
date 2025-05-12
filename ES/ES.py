#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 21:51:03 2025

@author: yonradeelimsawat
"""

import math
import random
import matplotlib.pyplot as plt

class EvolutionStrategy:
    """
    Implements a basic (μ + λ)-Evolution Strategy for optimizing the Rastrigin function
    with mutation probability and a population of N = 20.
    """
    
    def __init__(self, pop_size=20, mutation_prob=0.1, mutation_value=20, generations=100): #adjust generation and mutation value as needed  to see result
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.mutation_value = mutation_value
        self.generations = generations
        self.bounds = (-5.12, 5.12)
        self.dimension = 2
        self.population = self.initialize_population()
        self.history = []
        
        
    def initialize_population(self):
        """Initializes a population of N individuals within the defined bounds."""
        return [
            [random.uniform(*self.bounds) for _ in range(self.dimension)]
            for _ in range(self.pop_size)
        ]

    def rastrigin(self, x):
        """Rastrigin function as the fitness function."""
        A = 10
        return A * self.dimension + sum([(xi**2 - A * math.cos(2 * math.pi * xi)) for xi in x])

    def mutate(self, individual):
        """Mutates an individual with a fixed mutation probability and value."""
        new_individual = individual[:]
        for i in range(len(new_individual)):
            if random.random() < self.mutation_prob:
                change = random.choice([-1, 1]) * self.mutation_value * 0.01  # scale to ~±0.3
                new_individual[i] += change
                # Keep within bounds
                new_individual[i] = max(self.bounds[0], min(self.bounds[1], new_individual[i]))
        return new_individual

    def evolve(self):
        for gen in range(self.generations):
            offspring = [self.mutate(ind) for ind in self.population]
            combined = self.population + offspring
            combined.sort(key=self.rastrigin)  # sort by fitness (lower is better)
            self.population = combined[:self.pop_size]  # keep best N individuals
            best = self.rastrigin(self.population[0])
            self.history.append(best)
            print(f"Generation {gen+1}: Best Fitness = {best:.4f}")

    def display_result(self):
        best_solution = self.population[0]
        print("\nFinal Result")
        print(f"Best Solution: {best_solution}")
        print(f"Best Fitness: {self.rastrigin(best_solution):.4f}")

        # Plot convergence
        plt.plot(self.history)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Convergence of ES on Rastrigin Function")
        plt.grid(True)
        plt.show()

# Run the ES
if __name__ == '__main__':
    es = EvolutionStrategy()
    es.evolve()
    es.display_result()
        
