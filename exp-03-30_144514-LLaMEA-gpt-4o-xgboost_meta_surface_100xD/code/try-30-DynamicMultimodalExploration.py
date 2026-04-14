import numpy as np

class DynamicMultimodalExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = min(50, self.budget // 5)
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.adaptation_threshold = 0.05
        self.dynamic_resize_factor = 1.1
        self.elitism_factor = 0.1

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        return fitness

    def select_best(self, population, fitness):
        idx = np.argsort(fitness)
        num_elites = max(1, int(self.elitism_factor * len(population)))
        return population[idx][:num_elites]

    def differential_evolution(self, population, bounds):
        offspring = []
        for i in range(len(population)):
            x_t = population[i]
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            child = np.where(cross_points, mutant, x_t)
            offspring.append(child)
        return np.array(offspring)

    def fitness_scaling(self, fitness):
        scaled_fitness = fitness - np.min(fitness) + 1e-6
        return scaled_fitness

    def adapt_population_diversity(self, fitness):
        fitness_std = np.std(fitness)
        if fitness_std < self.adaptation_threshold:
            self.mutation_factor = min(self.mutation_factor * 1.2, 2.0)
            self.crossover_rate = min(self.crossover_rate * 1.1, 1.0)
            self.population_size = int(min(self.initial_population_size * self.dynamic_resize_factor, len(fitness) * self.dynamic_resize_factor))
        else:
            self.mutation_factor = max(self.mutation_factor * 0.8, 0.4)
            self.crossover_rate = max(self.crossover_rate * 0.9, 0.5)
            self.population_size = int(max(self.initial_population_size / self.dynamic_resize_factor, len(fitness) / self.dynamic_resize_factor))

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        best_solution = None
        best_fitness = np.inf

        while self.evaluations < self.budget:
            fitness = self.evaluate_population(population, func)
            scaled_fitness = self.fitness_scaling(fitness)
            self.adapt_population_diversity(scaled_fitness)
            elites = self.select_best(population, scaled_fitness)
            parents = np.concatenate((elites, population))
            population = self.differential_evolution(parents, bounds)

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx]

        return best_solution, best_fitness