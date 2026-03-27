import numpy as np

class EnhancedChaoticIslandOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_min = 0.5  
        self.F_max = 0.9  
        self.CR_min = 0.2  
        self.CR_max = 0.9  
        self.population = None
        self.fitness = None
        self.eval_count = 0
        self.island_count = 5
        self.migration_interval = 100

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.island_count, self.population_size // self.island_count, self.dim))
        self.fitness = np.full((self.island_count, self.population_size // self.island_count), np.inf)

    def chaotic_map(self, x):
        return np.sin(np.pi * x)

    def adaptive_mutation(self, target_idx, island_idx, bounds, F):
        indices = [idx for idx in range(self.population_size // self.island_count) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[island_idx, a] + F * (self.population[island_idx, b] - self.population[island_idx, c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def adapt_params(self):
        chaos_value = self.chaotic_map(np.random.rand())
        F = self.F_min + chaos_value * (self.F_max - self.F_min)
        CR = self.CR_min + chaos_value * (self.CR_max - self.CR_min)
        return F, CR

    def update_population(self, func, bounds):
        for island_idx in range(self.island_count):
            for i in range(self.population_size // self.island_count):
                if self.eval_count >= self.budget:
                    break
                F, CR = self.adapt_params()
                mutant = self.adaptive_mutation(i, island_idx, bounds, F)
                trial = self.crossover(self.population[island_idx, i], mutant, CR)
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < self.fitness[island_idx, i]:
                    self.population[island_idx, i] = trial
                    self.fitness[island_idx, i] = trial_fitness

    def migrate_individuals(self):
        for island_idx in range(self.island_count):
            best_idx = np.argmin(self.fitness[island_idx])
            best_individual = self.population[island_idx, best_idx]
            target_island = (island_idx + 1) % self.island_count
            worst_idx = np.argmax(self.fitness[target_island])
            self.population[target_island, worst_idx] = best_individual
            self.fitness[target_island, worst_idx] = self.fitness[island_idx, best_idx]

    def dynamic_clustering_and_restart(self, bounds):
        if self.eval_count % self.migration_interval == 0:
            self.migrate_individuals()
        if self.eval_count >= self.budget * 0.8:
            self.initialize_population(bounds)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)

        while self.eval_count < self.budget:
            self.update_population(func, bounds)
            self.dynamic_clustering_and_restart(bounds)

        best_island_idx, best_ind_idx = np.unravel_index(np.argmin(self.fitness, axis=None), self.fitness.shape)
        return self.population[best_island_idx, best_ind_idx], self.fitness[best_island_idx, best_ind_idx]