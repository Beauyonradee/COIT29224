#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 12:49:10 2025

@author: yonradeelimsawat
"""

import math
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, pop_size=20, mutation_prob=0.1, crossover_prob=0.8, mutation_value=5, generations=100):
        # Initialize parameters for the Genetic Algorithm
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.mutation_value = mutation_value
        self.generations = generations
        self.bounds = (-5.12, 5.12)
        self.dimension = 2
        self.population = self.initialize_population()
        self.history = []

    def initialize_population(self):
        # Create a population with random values within the bounds
        return [
            [random.uniform(*self.bounds) for _ in range(self.dimension)]
            for _ in range(self.pop_size)
        ]

    def rastrigin(self, x):
        # Rastrigin function (benchmark function with many local minima)
        A = 10
        return A * self.dimension + sum([(xi**2 - A * math.cos(2 * math.pi * xi)) for xi in x])

    def select_parents(self):
        # Select 2 best parents out of 4 randomly picked individuals (tournament selection)
        return sorted(random.sample(self.population, 4), key=self.rastrigin)[:2]

    def crossover(self, parent1, parent2):
        # Perform single-point crossover with some probability
        if random.random() < self.crossover_prob:
            point = random.randint(1, self.dimension - 1)
            return parent1[:point] + parent2[point:]
        return parent1[:] # No crossover, return copy of parent1

    def mutate(self, individual):
        # Randomly mutate each gene (dimension) with some probability
        for i in range(self.dimension):
            if random.random() < self.mutation_prob:
                change = random.choice([-1, 1]) * self.mutation_value * 0.01
                individual[i] += change
                # Keep the new value within bounds
                individual[i] = max(self.bounds[0], min(self.bounds[1], individual[i]))
        return individual

    def evolve(self):
        # Run the genetic algorithm for the specified number of generations
        for gen in range(self.generations):
            new_population = []
            while len(new_population) < self.pop_size:
                # Select parents and generate a child
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            # Sort by fitness and update population
            self.population = sorted(new_population, key=self.rastrigin)
            
            # Save the best fitness of this generation
            best = self.rastrigin(self.population[0])
            self.history.append(best)
            print(f"Generation {gen+1}: Best Fitness = {best:.4f}")

    def display_result(self):
        best_solution = self.population[0]
        print("\nFinal Result (GA)")
        print(f"Best Solution: {best_solution}")
        print(f"Best Fitness: {self.rastrigin(best_solution):.4f}")

        plt.plot(self.history, label="GA")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Convergence of Genetic Algorithm on Rastrigin Function")
        plt.grid(True)
        plt.legend()
        plt.show()

# Run the GA
if __name__ == '__main__':
    ga = GeneticAlgorithm()
    ga.evolve()
    ga.display_result()
