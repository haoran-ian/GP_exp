import numpy as np

class EnhancedAdaptiveMultimodalExploration:
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
        self.elitism_rate = 0.1  # New: Rate for elitism
        self.no_improve_count = 0  # New: Counter for stagnation
        self.restart_threshold = 10  # New: Threshold for adaptive restart

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        return fitness

    def select_best(self, population, fitness):
        idx = np.argsort(fitness)
        elites = int(self.population_size * self.elitism_rate)  # New: Calculate number of elites
        return population[idx][:self.population_size // 2], population[idx][:elites]  # Modified: Return elites

    def differential_evolution(self, population, bounds, elites=None):  # Modified: Optional elites parameter
        if elites is not None:  # New: Incorporate elites
            population = np.concatenate((population, elites))
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

    def stochastic_tunneling(self, fitness):
        min_fit = np.min(fitness)
        adjusted_fitness = np.exp(-fitness + min_fit)
        return adjusted_fitness

    def adapt_population_diversity(self, fitness):
        fitness_std = np.std(fitness)
        if fitness_std < self.adaptation_threshold:
            self.mutation_factor = min(self.mutation_factor * 1.2, 2.0)
            self.crossover_rate = min(self.crossover_rate * 1.1, 1.0)  # Adjust crossover rate
            self.population_size = int(min(self.initial_population_size * self.dynamic_resize_factor, len(fitness) * self.dynamic_resize_factor))
            self.no_improve_count += 1  # New: Increase stagnation counter
            if self.no_improve_count >= self.restart_threshold:  # New: Check for restart condition
                self.no_improve_count = 0
                return True  # New: Signal to restart
        else:
            self.mutation_factor = max(self.mutation_factor * 0.8, 0.4)
            self.crossover_rate = max(self.crossover_rate * 0.9, 0.5)  # Adjust crossover rate
            self.population_size = int(max(self.initial_population_size / self.dynamic_resize_factor, len(fitness) / self.dynamic_resize_factor))
            self.no_improve_count = 0  # New: Reset stagnation counter
        return False  # New: No restart needed

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        while self.evaluations < self.budget:
            fitness = self.evaluate_population(population, func)
            adjusted_fitness = self.stochastic_tunneling(fitness)
            restart = self.adapt_population_diversity(adjusted_fitness)  # Modified: Check restart
            parents, elites = self.select_best(population, adjusted_fitness)  # Modified: Get elites
            population = self.differential_evolution(parents, bounds, elites)  # Modified: Use elites
            if restart:  # New: Restart if condition is met
                population = self.initialize_population(bounds)
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]