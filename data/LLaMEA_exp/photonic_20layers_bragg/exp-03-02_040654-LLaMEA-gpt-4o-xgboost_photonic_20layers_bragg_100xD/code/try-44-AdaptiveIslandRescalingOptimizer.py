import numpy as np

class AdaptiveIslandRescalingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.fitness = None
        self.eval_count = 0
        self.island_count = 5
        self.migration_interval = 100
        self.memory = np.full(self.island_count, 0.5)  # Memory for mutation intensity

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.island_count, self.population_size // self.island_count, self.dim))
        self.fitness = np.full((self.island_count, self.population_size // self.island_count), np.inf)

    def adaptive_mutation(self, target_idx, island_idx, bounds):
        indices = [idx for idx in range(self.population_size // self.island_count) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F_intensity = self.memory[island_idx]
        mutant = self.population[island_idx, a] + self.F * np.random.uniform(0.9, 1.1) * F_intensity * (self.population[island_idx, b] - self.population[island_idx, c])
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
                mutant = self.adaptive_mutation(i, island_idx, bounds)
                trial = self.crossover(self.population[island_idx, i], mutant)
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < self.fitness[island_idx, i]:
                    self.population[island_idx, i] = trial
                    self.fitness[island_idx, i] = trial_fitness
                    # Update memory based on success
                    self.memory[island_idx] = min(1.0, self.memory[island_idx] + 0.1)
                else:
                    # Decrease the mutation intensity slightly if no improvement
                    self.memory[island_idx] = max(0.1, self.memory[island_idx] - 0.01)

    def migrate_individuals(self):
        for island_idx in range(self.island_count):
            best_idx = np.argmin(self.fitness[island_idx])
            best_individual = self.population[island_idx, best_idx]
            target_island = (island_idx + 1) % self.island_count
            worst_idx = np.argmax(self.fitness[target_island])
            # Replace the worst individual in the next island with the best from the current
            self.population[target_island, worst_idx] = best_individual
            self.fitness[target_island, worst_idx] = self.fitness[island_idx, best_idx]

    def dynamic_clustering_and_restart(self, bounds):
        if self.eval_count % self.migration_interval == 0:
            self.migrate_individuals()
        if self.eval_count >= self.budget * 0.8:
            # Rescale islands based on current performance
            performance = np.mean(self.fitness, axis=1)
            best_island_idx = np.argmin(performance)
            self.memory[:] = 0.5 + 0.5 * (performance - performance[best_island_idx]) / (np.max(performance) - performance[best_island_idx] + 1e-9)
            self.initialize_population(bounds)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)

        while self.eval_count < self.budget:
            self.update_population(func, bounds)
            self.dynamic_clustering_and_restart(bounds)

        best_island_idx, best_ind_idx = np.unravel_index(np.argmin(self.fitness, axis=None), self.fitness.shape)
        return self.population[best_island_idx, best_ind_idx], self.fitness[best_island_idx, best_ind_idx]