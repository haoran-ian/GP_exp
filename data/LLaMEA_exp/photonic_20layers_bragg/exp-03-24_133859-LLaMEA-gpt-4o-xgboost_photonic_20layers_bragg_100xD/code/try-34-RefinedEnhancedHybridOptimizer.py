import numpy as np

class RefinedEnhancedHybridOptimizer:
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
            return base_step * (0.5 ** (iteration / (self.budget // population_size)))

        def informed_local_search(ind, step_size=0.1, steps=12):
            candidate = ind.copy()
            for _ in range(steps):
                perturbation = np.random.uniform(-step_size, step_size, size=self.dim)
                candidate = np.clip(candidate + perturbation, lb, ub)
                candidate_fitness = func(candidate)
                if candidate_fitness < func(ind):
                    ind = candidate
                    break
            return ind

        def enhanced_differential_evolution():
            nonlocal evaluations
            for iteration in range(self.budget // population_size):
                success_count = 0
                diversity = np.std(population, axis=0).mean()
                for i in range(population_size):
                    if evaluations >= self.budget:
                        break
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    mut_factor = adaptive_step_size(0.7 + 0.3 * (1.0 - diversity), iteration)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    cross_prob = 0.8
                    cross_points = np.random.rand(self.dim) < cross_prob
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial
                        success_count += 1
                if success_count < population_size / 4:
                    break

        exploration_weight = 0.5
        for _ in range(self.budget // (population_size * 2)):
            enhanced_differential_evolution()
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                improved = informed_local_search(population[i], step_size=0.1 * exploration_weight)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
            exploration_weight *= 0.85  # Gradually reduce exploration over time

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]