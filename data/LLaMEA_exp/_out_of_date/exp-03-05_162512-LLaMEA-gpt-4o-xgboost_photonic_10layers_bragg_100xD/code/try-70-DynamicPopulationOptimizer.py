import numpy as np

class DynamicPopulationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = min(100, budget // 10)
        self.population_size = self.initial_population_size
        self.num_phases = 3
        self.phase_lengths = [self.budget // self.num_phases] * self.num_phases
        self.elite_archive = []

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        def initialize_population(size):
            return np.random.uniform(lb, ub, (size, self.dim))

        def evaluate_population(population):
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)
            return fitness

        def adaptive_exploitation_step(population, fitness):
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            offset = np.random.normal(loc=0, scale=0.1, size=(self.population_size, self.dim))
            neighbors = best_individual + offset
            neighbors = np.clip(neighbors, lb, ub)
            return neighbors

        def exploration_step(size):
            return np.random.uniform(lb, ub, (size, self.dim))

        def crossover(parent1, parent2):
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2

        def adjust_population_size():
            if self.evaluations < self.budget / 2:
                self.population_size = self.initial_population_size
            else:
                self.population_size = max(20, self.population_size // 2)

        population = initialize_population(self.population_size)
        fitness = evaluate_population(population)

        for phase in range(self.num_phases):
            if self.evaluations >= self.budget:
                break

            if phase == 0:  # Exploration
                population = exploration_step(self.population_size)
            elif phase == 1:  # Exploitation with crossover
                for _ in range(self.phase_lengths[phase]):
                    if self.evaluations >= self.budget:
                        break
                    parents = population[np.argsort(fitness)[:2]]
                    offspring = crossover(parents[0], parents[1])
                    offspring = np.clip(offspring, lb, ub)
                    offspring_fitness = func(offspring)
                    self.evaluations += 1
                    if offspring_fitness < max(fitness):
                        replace_idx = np.argmax(fitness)
                        population[replace_idx] = offspring
                        fitness[replace_idx] = offspring_fitness
            else:  # Adaptive exploitation
                population = adaptive_exploitation_step(population, fitness)

            fitness = evaluate_population(population)

            if len(self.elite_archive) < 5:
                self.elite_archive.append(population[np.argmin(fitness)])
            else:
                if func(population[np.argmin(fitness)]) < func(max(self.elite_archive, key=func)):
                    self.elite_archive.remove(max(self.elite_archive, key=func))
                    self.elite_archive.append(population[np.argmin(fitness)])

            adjust_population_size()

        best_idx = np.argmin(fitness)
        return population[best_idx] if not self.elite_archive else min(self.elite_archive, key=func)