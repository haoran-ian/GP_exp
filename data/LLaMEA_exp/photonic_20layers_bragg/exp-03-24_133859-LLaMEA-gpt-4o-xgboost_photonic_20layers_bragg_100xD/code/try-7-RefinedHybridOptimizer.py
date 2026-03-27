import numpy as np

class RefinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_fitness = np.min(fitness)

        def adaptive_params(fitness_variance):
            scale_factor = 0.5 + 0.3 * (1 - fitness_variance)
            crossover_rate = 0.7 + 0.2 * fitness_variance
            return scale_factor, crossover_rate

        def stochastic_local_search(ind, steps=5):
            candidate = ind.copy()
            for _ in range(steps):
                perturbation = np.random.normal(0, 0.1 * np.random.rand(), size=self.dim)
                candidate = np.clip(candidate + perturbation, lb, ub)
                if func(candidate) < func(ind):
                    ind = candidate
            return ind

        def adaptive_differential_evolution():
            for _ in range(self.budget // population_size):
                fitness_variance = np.var(fitness) / (np.mean(fitness) + 1e-6)
                scale_factor, crossover_rate = adaptive_params(fitness_variance)
                for i in range(population_size):
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    mutant = np.clip(x0 + scale_factor * (x1 - x2), lb, ub)
                    cross_points = np.random.rand(self.dim) < crossover_rate
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial

        for _ in range(self.budget // (population_size * 2)):
            adaptive_differential_evolution()
            for i in range(population_size):
                improved = stochastic_local_search(population[i])
                improved_fitness = func(improved)
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
                if improved_fitness < best_fitness:
                    best_fitness = improved_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]