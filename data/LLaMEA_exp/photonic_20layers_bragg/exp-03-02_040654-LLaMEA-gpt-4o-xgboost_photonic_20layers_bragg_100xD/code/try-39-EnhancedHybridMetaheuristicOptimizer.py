import numpy as np

class EnhancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.fitness = None
        self.eval_count = 0

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def adaptive_mutation(self, target_idx, bounds):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        # Change: Introduce a slight variation in the mutation factor for enhanced exploration.
        mutant = self.population[a] + self.F * np.random.uniform(0.9, 1.1) * (self.population[b] - self.population[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def update_population(self, func, bounds):
        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break
            mutant = self.adaptive_mutation(i, bounds)
            trial = self.crossover(self.population[i], mutant)
            trial_fitness = func(trial)
            self.eval_count += 1
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
        perturbation = np.random.uniform(-0.1, 0.1, cluster_centers.shape) * (bounds.ub - bounds.lb)
        phase_transitions = np.random.choice([-1, 1], size=cluster_centers.shape) * perturbation
        self.population = cluster_centers + phase_transitions
        self.fitness = np.full(len(self.population), np.inf)

    def dynamic_cluster_exploration(self, bounds):
        if self.population_size < 20 * self.dim:
            new_samples = np.random.uniform(bounds.lb, bounds.ub, (10 * self.dim, self.dim))
            self.population = np.vstack((self.population, new_samples))
            self.fitness = np.concatenate((self.fitness, np.full(new_samples.shape[0], np.inf)))
            self.population_size = len(self.population)

    def restart_strategy(self, bounds):
        if self.eval_count >= self.budget * 0.8 and self.population_size > 5 * self.dim:
            self.initialize_population(bounds)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)

        while self.eval_count < self.budget:
            self.update_population(func, bounds)
            if self.eval_count < self.budget:
                self.landscape_aware_selection()
                self.exploit_diverse_areas(bounds)
                self.dynamic_cluster_exploration(bounds)
                self.restart_strategy(bounds)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]