import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.8  # Differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 5
        self.temp = 1.0  # Initial temperature for simulated annealing

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while evaluations < self.budget:
            # Differential evolution - selection, mutation and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i] or np.exp((fitness[i] - trial_fit) / self.temp) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fit

                # Local search with dynamic scaling factor for exploitation
                dynamic_scale = (ub - lb) * (0.1 + 0.9 * (self.budget - evaluations) / self.budget)
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    neighbor = trial + np.random.uniform(-0.1, 0.1, self.dim) * dynamic_scale
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx]
                best_fitness = fitness[current_best_idx]

            # Simulated annealing temperature decrease
            self.temp *= 0.99

        return best_solution