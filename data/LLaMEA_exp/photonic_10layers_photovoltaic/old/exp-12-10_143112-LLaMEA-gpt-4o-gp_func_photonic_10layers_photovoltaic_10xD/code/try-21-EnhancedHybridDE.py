import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_base = 0.5  # Base differential mutation factor
        self.cr_base = 0.7  # Base crossover probability
        self.local_search_iters = 5
        self.dynamic_factor = 0.1  # Factor for adaptive parameter control

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive parameter control
            f = self.f_base + self.dynamic_factor * np.random.uniform(-0.1, 0.1)
            cr = self.cr_base + self.dynamic_factor * np.random.uniform(-0.1, 0.1)

            # Differential evolution - selection, mutation and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + f * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

                # Dynamic local search for exploitation
                local_search_intensity = int(self.local_search_iters * (1 - evaluations / self.budget))
                for _ in range(local_search_intensity):
                    if evaluations >= self.budget:
                        break

                    step_size = np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                    neighbor = trial + step_size
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

            # Update population with the best found in local search
            population[i] = trial
            fitness[i] = trial_fit

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]