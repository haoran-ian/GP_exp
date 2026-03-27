import numpy as np

class EnhancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.population = None
        self.fitness = None
        self.dynamic_F = lambda eval_ratio: 0.5 + 0.3 * np.sin(np.pi * eval_ratio)
        self.dynamic_CR = lambda eval_ratio: 0.6 + 0.2 * np.cos(np.pi * eval_ratio)

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def adaptive_mutation(self, target_idx, bounds, eval_ratio):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F_dynamic = self.dynamic_F(eval_ratio)
        mutant = self.population[a] + F_dynamic * (self.population[b] - self.population[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant, eval_ratio):
        CR_dynamic = self.dynamic_CR(eval_ratio)
        crossover_mask = np.random.rand(self.dim) < CR_dynamic
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def update_population(self, func, bounds, eval_ratio):
        for i in range(self.population_size):
            mutant = self.adaptive_mutation(i, bounds, eval_ratio)
            trial = self.crossover(self.population[i], mutant, eval_ratio)
            trial_fitness = func(trial)
            if trial_fitness < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness

    def landscape_aware_selection(self):
        ranked_indices = np.argsort(self.fitness)
        top_half_indices = ranked_indices[:self.population_size // 2]
        self.population = self.population[top_half_indices]
        self.fitness = self.fitness[top_half_indices]
        self.population_size //= 2

    def exploit_diverse_areas(self, bounds):
        cluster_size = min(5, len(self.population))
        clusters = np.random.choice(np.arange(len(self.population)), cluster_size, replace=False)
        cluster_centers = self.population[clusters]
        self.population = cluster_centers + np.random.uniform(-0.1, 0.1, cluster_centers.shape) * (bounds.ub - bounds.lb)
        self.fitness = np.full(len(self.population), np.inf)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        eval_count = 0

        while eval_count < self.budget:
            eval_ratio = eval_count / self.budget
            self.update_population(func, bounds, eval_ratio)
            eval_count += self.population_size

            if eval_count < self.budget:
                self.landscape_aware_selection()
                self.exploit_diverse_areas(bounds)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]