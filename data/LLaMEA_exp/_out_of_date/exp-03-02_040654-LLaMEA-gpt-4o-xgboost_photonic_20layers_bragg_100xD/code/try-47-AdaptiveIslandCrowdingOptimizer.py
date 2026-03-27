import numpy as np

class AdaptiveIslandCrowdingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.island_count = 5
        self.migration_interval = 100
        self.population = None
        self.fitness = None
        self.eval_count = 0
        self.island_sizes = None
        self.crowding_distances = None

    def initialize_population(self, bounds):
        self.island_sizes = np.random.randint(1, self.population_size // self.island_count, size=self.island_count)
        self.island_sizes = self.island_sizes / np.sum(self.island_sizes) * self.population_size
        self.island_sizes = self.island_sizes.astype(int)
        self.population = [np.random.uniform(bounds.lb, bounds.ub, (size, self.dim)) for size in self.island_sizes]
        self.fitness = [np.full(size, np.inf) for size in self.island_sizes]
        self.crowding_distances = [np.zeros(size) for size in self.island_sizes]

    def adaptive_mutation(self, target_idx, island_idx, bounds):
        size = self.island_sizes[island_idx]
        indices = [idx for idx in range(size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[island_idx][a] + self.F * np.random.uniform(0.9, 1.1) * (self.population[island_idx][b] - self.population[island_idx][c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def update_population(self, func, bounds):
        for island_idx, size in enumerate(self.island_sizes):
            for i in range(size):
                if self.eval_count >= self.budget:
                    break
                mutant = self.adaptive_mutation(i, island_idx, bounds)
                trial = self.crossover(self.population[island_idx][i], mutant)
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < self.fitness[island_idx][i]:
                    self.population[island_idx][i] = trial
                    self.fitness[island_idx][i] = trial_fitness

    def calculate_crowding_distances(self):
        for island_idx, size in enumerate(self.island_sizes):
            if size == 0:
                continue
            sorted_indices = np.argsort(self.fitness[island_idx])
            for i in range(size):
                if i == 0 or i == size - 1:
                    self.crowding_distances[island_idx][sorted_indices[i]] = float('inf')
                else:
                    self.crowding_distances[island_idx][sorted_indices[i]] = (
                        self.fitness[island_idx][sorted_indices[i + 1]] - self.fitness[island_idx][sorted_indices[i - 1]]
                    )

    def migrate_individuals(self):
        self.calculate_crowding_distances()
        for island_idx, size in enumerate(self.island_sizes):
            if size == 0:
                continue
            best_idx = np.argmin(self.fitness[island_idx] + self.crowding_distances[island_idx])
            best_individual = self.population[island_idx][best_idx]
            target_island = (island_idx + 1) % self.island_count
            worst_idx = np.argmax(self.fitness[target_island] - self.crowding_distances[target_island])
            # Replace the worst individual in the next island with the best from current
            self.population[target_island][worst_idx] = best_individual
            self.fitness[target_island][worst_idx] = self.fitness[island_idx][best_idx]

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

        best_island_idx, best_ind_idx = min(
            ((i, np.argmin(fitness)) for i, fitness in enumerate(self.fitness) if len(fitness) > 0),
            key=lambda x: self.fitness[x[0]][x[1]]
        )
        return self.population[best_island_idx][best_ind_idx], self.fitness[best_island_idx][best_ind_idx]