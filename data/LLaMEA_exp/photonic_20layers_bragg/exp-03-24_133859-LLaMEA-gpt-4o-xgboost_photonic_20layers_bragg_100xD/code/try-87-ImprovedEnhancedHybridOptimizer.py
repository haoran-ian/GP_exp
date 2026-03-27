import numpy as np

class ImprovedEnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        initial_population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = initial_population_size

        def adaptive_step_size(base_step, iteration, success_rate):
            adjustment = 1.0 + (0.5 * (success_rate - 0.5))
            return base_step * (0.5 ** (iteration / (self.budget // initial_population_size))) * adjustment

        def neighborhood_search(ind, step_size):
            perturbation = np.random.uniform(-step_size, step_size, size=self.dim)
            candidate = np.clip(ind + perturbation, lb, ub)
            return candidate

        def probabilistic_local_search(ind, step_size, probability):
            if np.random.rand() < probability:
                return neighborhood_search(ind, step_size)
            return ind

        def adaptive_differential_evolution():
            nonlocal evaluations
            for iteration in range(self.budget // initial_population_size):
                success_count = 0
                success_rate = 0.0
                for i in range(initial_population_size):
                    if evaluations >= self.budget:
                        return
                    idxs = np.random.choice(np.delete(np.arange(initial_population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    diversity_factor = np.mean(np.std(population, axis=0))
                    mut_factor = adaptive_step_size(0.8, iteration, success_rate)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    cross_prob = 0.9
                    trial = np.where(np.random.rand(self.dim) < cross_prob, mutant, population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial
                        success_count += 1
                success_rate = success_count / initial_population_size

        exploration_weight = 0.5
        for _ in range(self.budget // (initial_population_size * 2)):
            adaptive_differential_evolution()
            for i in range(initial_population_size):
                if evaluations >= self.budget:
                    return population[np.argmin(fitness)], fitness.min()
                improved = probabilistic_local_search(population[i], step_size=0.1 * exploration_weight, probability=0.5)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
            exploration_weight *= 0.9

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]