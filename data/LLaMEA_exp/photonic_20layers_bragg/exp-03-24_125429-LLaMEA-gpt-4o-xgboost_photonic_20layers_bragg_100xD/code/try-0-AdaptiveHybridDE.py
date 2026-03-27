import numpy as np

class AdaptiveHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.local_search_prob = 0.1  # Probability of applying local search
        self.adaptive_rate = 0.1  # Rate of adaptation for mutation strategy

    def _initialize_population(self, bounds):
        pop = np.random.rand(self.population_size, self.dim)
        return bounds.lb + pop * (bounds.ub - bounds.lb)

    def _mutate(self, pop, idx):
        a, b, c = np.random.choice(np.delete(np.arange(self.population_size), idx), 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def _local_search(self, candidate, bounds):
        perturbed = candidate + np.random.normal(0, 0.1, self.dim) * (bounds.ub - bounds.lb)
        return np.clip(perturbed, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        pop = self._initialize_population(bounds)
        best_idx = np.argmin([func(ind) for ind in pop])
        best = pop[best_idx]
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                mutant = self._mutate(pop, i)
                trial = self._crossover(pop[i], mutant)
                
                if np.random.rand() < self.local_search_prob:
                    trial = self._local_search(trial, bounds)
                
                trial_fitness = func(trial)
                if trial_fitness < func(pop[i]):
                    pop[i] = trial
                    if trial_fitness < func(best):
                        best = trial
                        
                # Adjust F and CR adaptively based on performance
                if trial_fitness < func(best):
                    self.F = min(1.0, self.F + self.adaptive_rate)
                    self.CR = min(1.0, self.CR + self.adaptive_rate)
                else:
                    self.F = max(0.5, self.F - self.adaptive_rate)
                    self.CR = max(0.1, self.CR - self.adaptive_rate)

        return best