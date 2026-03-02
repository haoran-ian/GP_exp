import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 5
        self.population = None
        self.fitness_values = None
        self.global_best = None
        self.global_best_value = np.inf

    def initialize(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness_values = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fitness = func(self.population[i])
            if fitness < self.fitness_values[i]:
                self.fitness_values[i] = fitness
            if fitness < self.global_best_value:
                self.global_best_value = fitness
                self.global_best = self.population[i].copy()

    def differential_evolution_step(self, lb, ub, evaluations):
        for i in range(self.population_size):
            idxs = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = self.population[idxs]
            mutation_factor = 0.5 + 0.5 * (1 - evaluations / self.budget)
            trial_vector = np.clip(x1 + mutation_factor * (x2 - x3), lb, ub)
            
            cross_probability = 0.7 + 0.3 * (evaluations / self.budget)
            crossover = np.random.rand(self.dim) < cross_probability
            offspring = np.where(crossover, trial_vector, self.population[i])
            
            offspring_fitness = func(offspring)
            if offspring_fitness < self.fitness_values[i]:
                self.fitness_values[i] = offspring_fitness
                self.population[i] = offspring
                if offspring_fitness < self.global_best_value:
                    self.global_best_value = offspring_fitness
                    self.global_best = offspring

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            self.differential_evolution_step(lb, ub, evaluations)
            evaluations += self.population_size

        return self.global_best_value, self.global_best