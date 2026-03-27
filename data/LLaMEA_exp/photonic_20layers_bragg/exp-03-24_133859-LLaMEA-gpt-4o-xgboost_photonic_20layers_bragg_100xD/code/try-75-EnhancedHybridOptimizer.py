import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = 0

        def adaptive_step_size(base_step, iteration):
            return base_step * (0.9 ** (iteration / (self.budget // population_size)))

        def neighborhood_search(ind, step_size=0.1):
            perturbation = np.random.normal(0, step_size, size=self.dim)
            candidate = np.clip(ind + perturbation, lb, ub)
            return candidate

        def memetic_local_improvement(ind, step_size=0.05, steps=5):
            best_candidate = ind.copy()
            best_candidate_fitness = func(ind)
            for _ in range(steps):
                candidate = neighborhood_search(best_candidate, step_size)
                candidate_fitness = func(candidate)
                if candidate_fitness < best_candidate_fitness:
                    best_candidate, best_candidate_fitness = candidate, candidate_fitness
            return best_candidate

        def self_adaptive_differential_evolution():
            nonlocal evaluations
            for iteration in range(self.budget // population_size):
                F_base = 0.5 + 0.3 * (np.sin(2 * np.pi * iteration / self.budget) + 1) / 2
                CR_base = 0.9 - 0.4 * (np.sin(2 * np.pi * iteration / self.budget) + 1) / 2
                for i in range(population_size):
                    if evaluations >= self.budget:
                        return
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    F = F_base * (1 + np.random.uniform(-0.1, 0.1))
                    mutant = np.clip(x0 + F * (x1 - x2), lb, ub)
                    CR = CR_base * (1 + np.random.uniform(-0.1, 0.1))
                    cross_points = np.random.rand(self.dim) < CR
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial

        for _ in range(self.budget // (population_size * 2)):
            self_adaptive_differential_evolution()
            for i in range(population_size):
                if evaluations >= self.budget:
                    return population[np.argmin(fitness)], fitness.min()
                improved = memetic_local_improvement(population[i], step_size=0.1)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]