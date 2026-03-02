import numpy as np

class EnhancedHierarchicalIslandOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5
        self.CR = 0.9
        self.population = None
        self.fitness = None
        self.eval_count = 0
        self.island_count = 5
        self.migration_interval = 50
        self.scaling_factor = 0.2

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.island_count, self.population_size // self.island_count, self.dim))
        self.fitness = np.full((self.island_count, self.population_size // self.island_count), np.inf)

    def self_adaptive_mutation(self, target_idx, island_idx, bounds):
        indices = [idx for idx in range(self.population_size // self.island_count) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = self.F_base * (1 + np.random.uniform(-self.scaling_factor, self.scaling_factor))
        mutant = self.population[island_idx, a] + F * (self.population[island_idx, b] - self.population[island_idx, c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def update_population(self, func, bounds):
        for island_idx in range(self.island_count):
            for i in range(self.population_size // self.island_count):
                if self.eval_count >= self.budget:
                    break
                mutant = self.self_adaptive_mutation(i, island_idx, bounds)
                trial = self.crossover(self.population[island_idx, i], mutant)
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < self.fitness[island_idx, i]:
                    self.population[island_idx, i] = trial
                    self.fitness[island_idx, i] = trial_fitness

    def hybrid_migration(self):
        for island_idx in range(self.island_count):
            best_idx = np.argmin(self.fitness[island_idx])
            best_individual = self.population[island_idx, best_idx]
            for target_island in range(self.island_count):
                if target_island != island_idx:
                    target_idx = np.argmax(self.fitness[target_island])
                    self.population[target_island, target_idx] = best_individual
                    self.fitness[target_island, target_idx] = self.fitness[island_idx, best_idx]

    def dynamic_island_scaling(self, bounds):
        if self.eval_count % self.migration_interval == 0:
            self.hybrid_migration()
            if self.eval_count >= self.budget * 0.7:
                new_island_count = np.random.randint(3, 8)
                self.island_count = new_island_count
                self.initialize_population(bounds)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)

        while self.eval_count < self.budget:
            self.update_population(func, bounds)
            self.dynamic_island_scaling(bounds)

        best_island_idx, best_ind_idx = np.unravel_index(np.argmin(self.fitness, axis=None), self.fitness.shape)
        return self.population[best_island_idx, best_ind_idx], self.fitness[best_island_idx, best_ind_idx]