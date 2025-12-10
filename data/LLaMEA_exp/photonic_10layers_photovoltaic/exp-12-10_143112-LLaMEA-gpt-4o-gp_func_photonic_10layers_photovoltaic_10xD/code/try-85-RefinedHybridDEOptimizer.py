import numpy as np

class RefinedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.8  # Differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 5
        self.elitism_rate = 0.1  # Proportion of elite individuals

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Differential evolution - selection, mutation and crossover
            new_population = []

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

                if trial_fit < fitness[i]:
                    new_population.append((trial, trial_fit))
                else:
                    new_population.append((population[i], fitness[i]))

                # Adaptive local search for exploitation
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    step_size = 0.1 * (ub - lb) / (1 + evaluations/self.budget)
                    neighbor = trial + np.random.uniform(-step_size, step_size, self.dim)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit
                        new_population[-1] = (trial, trial_fit)

            # Select the elite individuals
            new_population.sort(key=lambda x: x[1])
            elite_count = int(self.elitism_rate * self.population_size)
            population, fitness = zip(*new_population[:self.population_size - elite_count])

            # Add elite individuals
            elite_individuals = [(ind, fit) for ind, fit in new_population[:elite_count]]
            population = list(population) + [ind for ind, _ in elite_individuals]
            fitness = list(fitness) + [fit for _, fit in elite_individuals]

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]