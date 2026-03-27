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

        def neighborhood_search(ind, step_size=0.1):
            perturbation = np.random.uniform(-step_size, step_size, size=self.dim)
            candidate = np.clip(ind + perturbation, lb, ub)
            return candidate
        
        def stochastic_local_search(ind, step_size=0.1, steps=10):
            best_candidate = ind.copy()
            best_candidate_fitness = func(ind)
            for _ in range(steps):
                candidate = neighborhood_search(ind, step_size)
                candidate_fitness = func(candidate)
                if candidate_fitness < best_candidate_fitness:
                    best_candidate, best_candidate_fitness = candidate, candidate_fitness
            return best_candidate

        def adaptive_differential_evolution(population, fitness):
            nonlocal evaluations
            success_count = 0
            for iteration in range(self.budget // (2 * population_size)):
                if evaluations >= self.budget:
                    return
                diversity = np.std(population, axis=0).mean()
                diversity_factor = 1 / (1 + np.exp(-10 * (diversity - 0.5)))
                for i in range(population_size):
                    if evaluations >= self.budget:
                        return
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    mut_factor = adaptive_step_size(0.6 + 0.4 * diversity_factor, iteration)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    cross_prob = 0.8 * (1 - (success_count / (i+1) if i > 0 else 0))
                    cross_points = np.random.rand(self.dim) < cross_prob
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial
                        success_count += 1

        def dynamic_population_resizing():
            nonlocal population, fitness
            sorted_indices = np.argsort(fitness)
            top_individuals = population[sorted_indices[:population_size // 2]]
            new_individuals = np.random.uniform(lb, ub, (population_size // 2, self.dim))
            population = np.vstack((top_individuals, new_individuals))
            fitness = np.array([func(ind) for ind in population])
        
        exploration_weight = 0.5
        for _ in range(self.budget // (population_size * 3)):
            adaptive_differential_evolution(population, fitness)
            for i in range(population_size):
                if evaluations >= self.budget:
                    return population[np.argmin(fitness)], fitness.min()
                improved = stochastic_local_search(population[i], step_size=0.1 * exploration_weight)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
            exploration_weight *= 0.95
            dynamic_population_resizing()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]