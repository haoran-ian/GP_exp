import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        def local_search(ind, steps=10):
            candidate = ind.copy()
            for _ in range(steps):
                perturbation = np.random.normal(0, 0.1, size=self.dim)
                candidate = np.clip(candidate + perturbation, lb, ub)
                if func(candidate) < func(ind):
                    ind = candidate
            return ind

        def differential_evolution():
            nonlocal evaluations
            for _ in range(self.budget // population_size):
                for i in range(population_size):
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    mutant = np.clip(x0 + 0.8 * (x1 - x2), lb, ub)
                    cross_points = np.random.rand(self.dim) < 0.9
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial

        while evaluations < self.budget:
            differential_evolution()
            for i in range(population_size):
                improved = local_search(population[i])
                new_fitness = func(improved)
                evaluations += 1
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    population[i] = improved

            # Dynamic population resizing based on fitness variance
            fitness_var = np.var(fitness)
            if fitness_var < 0.01:
                population_size = max(5 * self.dim, int(0.8 * population_size))
            else:
                population_size = min(20 * self.dim, int(1.2 * population_size))
            population = population[:population_size]
            fitness = fitness[:population_size]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]