import numpy as np

class EnhancedCMADEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.local_search_perturbation = 0.05
        self.adaptation_rate = 0.2
        self.cov_matrix = np.eye(dim)  # Initialize with identity matrix

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            trial_population = np.empty_like(population)
            fitness_variance = np.var(fitness)
            max_fitness_diff = np.max(fitness) - np.min(fitness)
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Fitness-weighted Differential Evolution Mutation with Covariance Matrix Adaptation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                weight = (fitness[indices[0]] - fitness[i]) / (1e-9 + max_fitness_diff)
                direction = np.dot(np.random.multivariate_normal(np.zeros(self.dim), self.cov_matrix), b - c)
                mutant = np.clip(a + weight * self.mutation_factor * direction, lb, ub)

                # Adaptive Crossover with dynamic adjustment
                self.crossover_probability = 0.9 - (fitness_variance / (1e-9 + np.max(fitness_variance))) * 0.5
                crossover = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                evals += 1

                # Selection with covariance adaptation
                if trial_fitness < fitness[i]:
                    trial_population[i] = trial
                    fitness[i] = trial_fitness
                    self.mutation_factor = min(1.0, self.mutation_factor + self.adaptation_rate * 1.1)
                    # Update covariance matrix
                    delta = trial - population[i]
                    self.cov_matrix += np.outer(delta, delta) / self.population_size
                else:
                    trial_population[i] = population[i]
                    self.mutation_factor = max(0.1, self.mutation_factor - self.adaptation_rate)

                # Adaptive Stochastic Local Search
                if evals < self.budget:
                    perturbation = self.local_search_perturbation + 0.01 * fitness_variance
                    local_trial = trial + perturbation * np.random.multivariate_normal(np.zeros(self.dim), self.cov_matrix)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evals += 1
                    if local_fitness < fitness[i]:
                        trial_population[i] = local_trial
                        fitness[i] = local_fitness
                        # Update covariance matrix
                        delta = local_trial - trial_population[i]
                        self.cov_matrix += np.outer(delta, delta) / self.population_size

            population[:] = trial_population

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]