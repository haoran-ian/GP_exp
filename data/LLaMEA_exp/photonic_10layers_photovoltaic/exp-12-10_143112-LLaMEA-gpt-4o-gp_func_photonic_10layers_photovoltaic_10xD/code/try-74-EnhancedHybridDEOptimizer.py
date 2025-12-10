import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.f = 0.8  # Differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 5
        self.population_shrink_rate = 0.95

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit
                
                local_search_improved = False
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    step_size = 0.1 * (ub - lb) * (1 - evaluations / self.budget)
                    neighbor = trial + np.random.uniform(-step_size, step_size, self.dim)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit
                        local_search_improved = True

                # Apply new trial to population if improved by local search
                if local_search_improved or trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

            # Shrink population size over time to enhance exploitation
            population_size = max(2, int(self.initial_population_size * (self.population_shrink_rate ** (evaluations / self.budget))))
            population = population[:population_size]
            fitness = fitness[:population_size]

        best_idx = np.argmin(fitness)
        return population[best_idx]