import numpy as np

class DynamicAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        def adaptive_step_size(base_step, iteration, max_iterations):
            phase_progress = iteration / max_iterations
            phase_variation = 0.3 * np.sin(2 * np.pi * phase_progress)
            return base_step * (0.5 ** (iteration / max_iterations)) * (1 + phase_variation)

        def neighborhood_search(ind, step_size=0.1):
            candidate = ind.copy()
            perturbation = np.random.uniform(-step_size, step_size, size=self.dim)
            candidate = np.clip(candidate + perturbation, lb, ub)
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

        def adaptive_differential_evolution(max_iterations):
            nonlocal evaluations
            for iteration in range(max_iterations):
                if evaluations >= self.budget:
                    return
                success_count = 0
                diversity = np.std(population, axis=0).mean()
                diversity_factor = np.tanh(diversity)  # Smooth scaling with hyperbolic tangent
                for i in range(population_size):
                    if evaluations >= self.budget:
                        return
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    mut_factor = adaptive_step_size(0.8 + 0.5 * diversity_factor, iteration, max_iterations)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    fitness_improvement_ratio = success_count / (i + 1) if i > 0 else 0
                    cross_prob = 0.9 * (1 - fitness_improvement_ratio)
                    cross_points = np.random.rand(self.dim) < cross_prob
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial
                        success_count += 1

        max_iterations = self.budget // (population_size * 2)
        exploration_weight = 0.5
        for iteration in range(max_iterations):
            adaptive_differential_evolution(max_iterations)
            for i in range(population_size):
                if evaluations >= self.budget:
                    return population[np.argmin(fitness)], fitness.min()
                step_size = 0.1 * exploration_weight
                improved = stochastic_local_search(population[i], step_size=step_size)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
            exploration_weight *= 0.9  # Gradually reduce exploration over time

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]