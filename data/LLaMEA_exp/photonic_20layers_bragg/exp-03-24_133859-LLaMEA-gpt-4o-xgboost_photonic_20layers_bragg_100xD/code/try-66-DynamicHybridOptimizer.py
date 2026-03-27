import numpy as np

class DynamicHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        def adaptive_mutation_factor(base_factor, diversity):
            return base_factor * (1 + 0.1 * np.tanh(diversity - 0.5))

        def neighborhood_search(ind, step_size):
            perturbation = np.random.uniform(-step_size, step_size, size=self.dim)
            return np.clip(ind + perturbation, lb, ub)

        def stochastic_local_search(ind, step_size, steps):
            best_candidate, best_fitness = ind.copy(), func(ind)
            for _ in range(steps):
                candidate = neighborhood_search(ind, step_size)
                candidate_fitness = func(candidate)
                if candidate_fitness < best_fitness:
                    best_candidate, best_fitness = candidate, candidate_fitness
            return best_candidate

        def dynamic_differential_evolution():
            nonlocal evaluations
            for iteration in range(self.budget // population_size):
                success_count = 0
                diversity = np.std(population, axis=0).mean()
                for i in range(population_size):
                    if evaluations >= self.budget:
                        return
                    idxs = np.random.choice(np.delete(np.arange(population_size), i), 4, replace=False)
                    x0, x1, x2, x3 = population[idxs]
                    mut_factor = adaptive_mutation_factor(0.8, diversity)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2 + x3 - x0), lb, ub)
                    cross_prob = 0.9 * (1 - (success_count / (i + 1)))
                    cross_points = np.random.rand(self.dim) < cross_prob
                    trial = np.where(cross_points, mutant, population[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial
                        success_count += 1

        exploration_weight = 0.5
        while evaluations < self.budget:
            dynamic_differential_evolution()
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                step_size = 0.1 * exploration_weight
                improved = stochastic_local_search(population[i], step_size, steps=5)
                improved_fitness = func(improved)
                evaluations += 1
                if improved_fitness < fitness[i]:
                    fitness[i] = improved_fitness
                    population[i] = improved
            exploration_weight *= 0.95

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]