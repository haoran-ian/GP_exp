import numpy as np

class DualPopulationEAME:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = min(50, self.budget // 5)
        self.exploratory_population_size = self.initial_population_size // 2
        self.exploitative_population_size = self.initial_population_size // 2
        self.mutation_factor = 0.9
        self.crossover_rate = 0.9
        self.adaptation_threshold = 0.05
        self.dynamic_resize_factor = 1.1

    def initialize_population(self, size, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        return fitness

    def select_best(self, population, fitness, size):
        idx = np.argsort(fitness)
        return population[idx][:size]

    def differential_evolution(self, population, bounds, mutation_factor, crossover_rate):
        offspring = []
        for i in range(len(population)):
            x_t = population[i]
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mutation_factor * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            child = np.where(cross_points, mutant, x_t)
            offspring.append(child)
        return np.array(offspring)

    def stochastic_tunneling(self, fitness):
        min_fit = np.min(fitness)
        adjusted_fitness = np.exp(-fitness + min_fit)
        return adjusted_fitness

    def adapt_population_diversity(self, fitness, pop_size):
        fitness_std = np.std(fitness)
        if fitness_std < self.adaptation_threshold:
            self.mutation_factor = min(self.mutation_factor * 1.2, 2.0)
            self.crossover_rate = min(self.crossover_rate * 1.1, 1.0)
            pop_size = int(pop_size * self.dynamic_resize_factor)
        else:
            self.mutation_factor = max(self.mutation_factor * 0.8, 0.4)
            self.crossover_rate = max(self.crossover_rate * 0.9, 0.5)
            pop_size = int(pop_size / self.dynamic_resize_factor)
        return pop_size

    def __call__(self, func):
        bounds = func.bounds
        exploratory_population = self.initialize_population(self.exploratory_population_size, bounds)
        exploitative_population = self.initialize_population(self.exploitative_population_size, bounds)
        best_solution = None
        best_fitness = np.inf

        while self.evaluations < self.budget:
            exploratory_fitness = self.evaluate_population(exploratory_population, func)
            exploitative_fitness = self.evaluate_population(exploitative_population, func)

            adjusted_exploratory_fitness = self.stochastic_tunneling(exploratory_fitness)
            adjusted_exploitative_fitness = self.stochastic_tunneling(exploitative_fitness)

            self.exploratory_population_size = self.adapt_population_diversity(adjusted_exploratory_fitness, self.exploratory_population_size)
            self.exploitative_population_size = self.adapt_population_diversity(adjusted_exploitative_fitness, self.exploitative_population_size)

            exploratory_parents = self.select_best(exploratory_population, adjusted_exploratory_fitness, self.exploratory_population_size)
            exploitative_parents = self.select_best(exploitative_population, adjusted_exploitative_fitness, self.exploitative_population_size)

            if np.min(exploratory_fitness) < best_fitness:
                best_solution = exploratory_population[np.argmin(exploratory_fitness)]
                best_fitness = np.min(exploratory_fitness)

            if np.min(exploitative_fitness) < best_fitness:
                best_solution = exploitative_population[np.argmin(exploitative_fitness)]
                best_fitness = np.min(exploitative_fitness)

            exploratory_population = self.differential_evolution(exploratory_parents, bounds, self.mutation_factor, self.crossover_rate)
            exploitative_population = self.differential_evolution(exploitative_parents, bounds, self.mutation_factor * 0.5, self.crossover_rate * 1.2)

        return best_solution, best_fitness