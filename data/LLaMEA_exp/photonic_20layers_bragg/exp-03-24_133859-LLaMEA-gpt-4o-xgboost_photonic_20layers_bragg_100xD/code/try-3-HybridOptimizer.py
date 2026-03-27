import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        def local_search(ind, steps=10):
            candidate = ind.copy()
            for _ in range(steps):
                perturbation = np.random.normal(0, 0.1, size=self.dim)
                candidate = np.clip(candidate + perturbation, lb, ub)
                if func(candidate) < func(ind):
                    ind = candidate
            return ind

        def differential_evolution():
            for _ in range(self.budget // population_size):
                for i in range(population_size):
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    F = 0.5 + np.random.rand() * 0.5  # Adaptive mutation scaling factor
                    mutant = np.clip(x0 + F * (x1 - x2), lb, ub)
                    cross_points = np.random.rand(self.dim) < 0.9
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial

        for _ in range(self.budget // (population_size * 2)):
            differential_evolution()
            for i in range(population_size):
                improved = local_search(population[i])
                if func(improved) < fitness[i]:
                    fitness[i] = func(improved)
                    population[i] = improved
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]