import numpy as np

class QuantumAdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.evaluations = 0
        self.q = 0.5  # Quantum probability
        self.chaotic_sequence = self.init_chaotic_sequence(self.population_size)

    def init_chaotic_sequence(self, size):
        # Initialize using a logistic map
        r = 3.9  # Chaotic parameter
        x0 = 0.5  # Initial value
        sequence = [x0]
        for _ in range(1, size):
            x0 = r * x0 * (1 - x0)
            sequence.append(x0)
        return np.array(sequence)

    def initialize_population(self, bounds):
        return [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(self.population_size)]

    def evaluate_population(self, population, func):
        return [func(ind) for ind in population]

    def select_parents(self, population, fitness):
        idx = np.random.choice(np.arange(len(population)), size=2, replace=False)
        return population[idx[0]] if fitness[idx[0]] < fitness[idx[1]] else population[idx[1]]

    def quantum_behavior(self, particle, global_best, bounds):
        u = np.random.rand(self.dim)
        beta = 1.5 * self.chaotic_sequence[self.evaluations % self.population_size]
        L = np.random.laplace(size=self.dim)
        new_position = particle + beta * (global_best - particle) * L
        return np.clip(new_position, bounds.lb, bounds.ub)

    def swarm_update(self, population, global_best, bounds):
        for i in range(len(population)):
            global_influence = self.quantum_behavior(population[i], global_best, bounds)
            if np.random.rand() < self.q:
                population[i] = global_influence
            else:
                inertia_weight = 0.9 - 0.5 * (self.evaluations / self.budget)
                velocity = np.random.rand(self.dim) * (global_best - population[i])
                velocity *= inertia_weight
                population[i] += velocity
                population[i] = np.clip(population[i], bounds.lb, bounds.ub)
        return population

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(population, func)
        best_individual = population[np.argmin(fitness)]
        best_fitness = min(fitness)
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.select_parents(population, fitness)
                parent2 = self.select_parents(population, fitness)
                new_individual = self.quantum_behavior(parent1, parent2, bounds)
                new_population.append(new_individual)

            new_fitness = self.evaluate_population(new_population, func)
            self.evaluations += len(new_population)

            combined_population = population + new_population
            combined_fitness = fitness + new_fitness

            selected_indices = np.argsort(combined_fitness)[:self.population_size]
            population = [combined_population[i] for i in selected_indices]
            fitness = [combined_fitness[i] for i in selected_indices]

            current_best = min(fitness)
            if current_best < best_fitness:
                best_fitness = current_best
                best_individual = population[np.argmin(fitness)]

            population = self.swarm_update(population, best_individual, bounds)

        return best_individual, best_fitness