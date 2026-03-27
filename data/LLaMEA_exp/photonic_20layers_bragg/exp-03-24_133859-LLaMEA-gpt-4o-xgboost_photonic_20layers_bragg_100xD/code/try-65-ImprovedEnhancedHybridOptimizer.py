import numpy as np

class ImprovedEnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        def adaptive_step_size(base_step, iteration):
            return base_step * (0.5 ** (iteration / (self.budget // population_size)))

        def chaotic_local_search(ind, step_size=0.1, steps=10):
            best_candidate = ind.copy()
            best_candidate_fitness = func(ind)
            z = np.random.uniform(0, 1)  # Logistic map initial value
            for _ in range(steps):
                z = 4 * z * (1 - z)  # Logistic map update
                perturbation = (np.random.rand(self.dim) - 0.5) * step_size * z
                candidate = np.clip(ind + perturbation, lb, ub)
                candidate_fitness = func(candidate)
                if candidate_fitness < best_candidate_fitness:
                    best_candidate, best_candidate_fitness = candidate, candidate_fitness
            return best_candidate

        def crowding_preservation(current_pop, new_pop, current_fit, new_fit):
            combined_pop = np.concatenate((current_pop, new_pop))
            combined_fit = np.concatenate((current_fit, new_fit))
            indices = np.argsort(combined_fit)
            return combined_pop[indices[:population_size]], combined_fit[indices[:population_size]]

        def adaptive_differential_evolution():
            nonlocal evaluations
            trial_population = np.empty_like(population)
            trial_fitness = np.full(population_size, np.inf)
            for iteration in range(self.budget // population_size):
                diversity = np.std(population, axis=0).mean()
                diversity_factor = 1 / (1 + np.exp(-10 * (diversity - 0.5)))
                for i in range(population_size):
                    if evaluations >= self.budget:
                        return
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    mut_factor = adaptive_step_size(0.8 + (0.5 * diversity_factor), iteration)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    cross_prob = 0.9 * (1 - (fitness[i] - fitness.min()) / (fitness.max() - fitness.min() + 1e-9))
                    cross_points = np.random.rand(self.dim) < cross_prob
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness[i] = func(trial)
                    evaluations += 1
                population, fitness = crowding_preservation(population, trial_population, fitness, trial_fitness)

        exploration_weight = 0.5
        for _ in range(self.budget // (population_size * 2)):
            adaptive_differential_evolution()
            for i in range(population_size):
                if evaluations >= self.budget:
                    return population[np.argmin(fitness)], fitness.min()
                improved = chaotic_local_search(population[i], step_size=0.1 * exploration_weight)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
            exploration_weight *= 0.9

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]