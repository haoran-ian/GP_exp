import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 10 * self.dim
        num_islands = 5  # Number of islands for the island model
        island_size = population_size // num_islands
        islands = [np.random.uniform(lb, ub, (island_size, self.dim)) for _ in range(num_islands)]
        fitnesses = [np.array([func(ind) for ind in island]) for island in islands]
        evaluations = 0

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

        def adaptive_differential_evolution(island, fitness):
            nonlocal evaluations
            for iteration in range(self.budget // population_size):
                success_count = 0
                diversity = np.std(island, axis=0).mean()
                diversity_factor = 1 / (1 + np.exp(-10 * (diversity - 0.5)))
                for i in range(island_size):
                    if evaluations >= self.budget:
                        return
                    idxs = np.random.choice(np.delete(np.arange(island_size), i), 3, replace=False)
                    x0, x1, x2 = island[idxs]
                    mut_factor = adaptive_step_size(0.8 + (0.5 * diversity_factor), iteration)
                    mutant = np.clip(x0 + mut_factor * (x1 - x2), lb, ub)
                    fitness_improvement_ratio = success_count / (i+1) if i > 0 else 0
                    cross_prob = 0.9 * (1 - fitness_improvement_ratio)
                    cross_points = np.random.rand(self.dim) < cross_prob
                    trial = np.where(cross_points, mutant, island[i])
                    trial_fitness = func(trial)
                    evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        island[i] = trial
                        success_count += 1

        exploration_weight = 0.5
        for _ in range(self.budget // (population_size * 2)):
            for island, fitness in zip(islands, fitnesses):
                adaptive_differential_evolution(island, fitness)
                for i in range(island_size):
                    if evaluations >= self.budget:
                        best_island_idx = np.argmin([fit.min() for fit in fitnesses])
                        best_idx = np.argmin(fitnesses[best_island_idx])
                        return islands[best_island_idx][best_idx], fitnesses[best_island_idx][best_idx]
                    improved = stochastic_local_search(island[i], step_size=0.1 * exploration_weight)
                    improved_fitness = func(improved)
                    evaluations += 1
                    if improved_fitness < fitness[i]:
                        fitness[i] = improved_fitness
                        island[i] = improved
            exploration_weight *= 0.9  # Reduce exploration over time
            # Migration step: exchange individuals between islands
            if evaluations < self.budget:
                for i in range(num_islands):
                    partner = (i + 1) % num_islands
                    swap_idx = np.random.randint(0, island_size)
                    islands[i][swap_idx], islands[partner][swap_idx] = islands[partner][swap_idx], islands[i][swap_idx]

        best_island_idx = np.argmin([fit.min() for fit in fitnesses])
        best_idx = np.argmin(fitnesses[best_island_idx])
        return islands[best_island_idx][best_idx], fitnesses[best_island_idx][best_idx]