import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.8  # Differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 5
        self.elitism_rate = 0.1  # Fraction of elitist solutions to preserve
        self.adaptive_rate = 0.05  # Rate of adaptation for f and cr

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Sort population by fitness for elitism
            elites_count = int(self.elitism_rate * self.population_size)
            sorted_indices = np.argsort(fitness)
            elites = population[sorted_indices[:elites_count]]

            # Differential evolution - mutation and crossover
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
                    new_population.append(trial)
                else:
                    new_population.append(population[i])

                # Local search for exploitation within trial
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    neighbor = trial + np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

            # Update population with elite preservation
            new_population = np.array(new_population)
            population = np.vstack((elites, new_population[elites_count:]))
            fitness = np.array([func(ind) for ind in population])

            # Adaptive parameter control
            self.f = np.clip(self.f + self.adaptive_rate * (np.random.rand() - 0.5), 0.5, 1.0)
            self.cr = np.clip(self.cr + self.adaptive_rate * (np.random.rand() - 0.5), 0.1, 1.0)

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]