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
            return base_step * (0.3 ** (iteration / (self.budget // population_size)))

        def neighborhood_search(ind, step_size=0.1):
            candidate = ind.copy()
            perturbation = np.random.randn(self.dim) * step_size
            candidate = np.clip(candidate + perturbation, lb, ub)
            return candidate

        def stochastic_local_search(ind, step_size=0.1, steps=15):
            best_candidate = ind.copy()
            best_candidate_fitness = func(ind)
            for _ in range(steps):
                candidate = neighborhood_search(ind, step_size)
                candidate_fitness = func(candidate)
                if candidate_fitness < best_candidate_fitness:
                    best_candidate, best_candidate_fitness = candidate, candidate_fitness
            return best_candidate

        def adaptive_differential_evolution():
            nonlocal evaluations
            for iteration in range(self.budget // population_size):
                success_count = 0
                diversity = np.std(population, axis=0).mean()
                diversity_factor = 1 / (1 + np.exp(-12 * (diversity - 0.3)))
                for i in range(population_size):
                    if evaluations >= self.budget:
                        return
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    mut_factor = adaptive_step_size(0.7 + (0.4 * diversity_factor), iteration)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    fitness_improvement_ratio = success_count / (i + 1) if i > 0 else 0
                    cross_prob = 0.8 * (1 - fitness_improvement_ratio)
                    cross_points = np.random.rand(self.dim) < cross_prob
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial
                        success_count += 1

        exploration_weight = 0.6
        for _ in range(self.budget // (population_size * 2)):
            adaptive_differential_evolution()
            for i in range(population_size):
                if evaluations >= self.budget:
                    return population[np.argmin(fitness)], fitness.min()
                improved = stochastic_local_search(population[i], step_size=0.1 * exploration_weight)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
            exploration_weight *= 0.85

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]