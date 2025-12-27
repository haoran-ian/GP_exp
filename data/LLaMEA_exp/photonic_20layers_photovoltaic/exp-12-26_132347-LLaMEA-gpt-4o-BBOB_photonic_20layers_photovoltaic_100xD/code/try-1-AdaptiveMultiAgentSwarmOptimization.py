import numpy as np

class AdaptiveMultiAgentSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.population = None
        self.best_agent = None
        self.evaluations = 0

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = [func(ind) for ind in self.population]
        self.best_agent = self.population[np.argmin(fitness)]

    def adaptive_parameters(self):
        # Adjust parameters based on the progress
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = np.random.uniform(0.7, 1.0)

    def differential_evolution(self, target_idx, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), lb, ub)

        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True

        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def local_search(self, agent, func):
        noise = np.random.normal(0, 0.1, self.dim)
        new_agent = np.clip(agent + noise, func.bounds.lb, func.bounds.ub)
        return new_agent

    def __call__(self, func):
        self.initialize_population(func)

        while self.evaluations < self.budget:
            self.adaptive_parameters()
            for idx in range(self.population_size):
                # Differential Evolution step
                trial = self.differential_evolution(idx, func)
                if func(trial) < func(self.population[idx]):
                    self.population[idx] = trial
                    self.evaluations += 1
                    if func(trial) < func(self.best_agent):
                        self.best_agent = trial

                # Local search step
                if self.evaluations < self.budget:
                    candidate = self.local_search(self.population[idx], func)
                    if func(candidate) < func(self.population[idx]):
                        self.population[idx] = candidate
                        self.evaluations += 1
                        if func(candidate) < func(self.best_agent):
                            self.best_agent = candidate

        return self.best_agent