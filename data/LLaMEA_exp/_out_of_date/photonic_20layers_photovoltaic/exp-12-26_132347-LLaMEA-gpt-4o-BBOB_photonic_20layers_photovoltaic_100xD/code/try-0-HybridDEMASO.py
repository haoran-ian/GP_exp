import numpy as np

class HybridDEMASO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.best_agent = None

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_agent = self.population[np.argmin([func(ind) for ind in self.population])]
    
    def differential_evolution(self, target_idx, bounds):
        lb, ub = bounds.lb, bounds.ub
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), lb, ub)

        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True

        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def local_search(self, agent, bounds):
        noise = np.random.normal(0, 0.1, self.dim)
        new_agent = np.clip(agent + noise, bounds.lb, bounds.ub)
        return new_agent

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        evaluations = 0

        while evaluations < self.budget:
            for idx in range(self.population_size):
                trial = self.differential_evolution(idx, bounds)
                if func(trial) < func(self.population[idx]):
                    self.population[idx] = trial
                    evaluations += 1
                    if func(trial) < func(self.best_agent):
                        self.best_agent = trial

                # Local search step
                if evaluations < self.budget:
                    candidate = self.local_search(self.population[idx], bounds)
                    if func(candidate) < func(self.population[idx]):
                        self.population[idx] = candidate
                        evaluations += 1
                        if func(candidate) < func(self.best_agent):
                            self.best_agent = candidate

        return self.best_agent