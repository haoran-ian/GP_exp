import numpy as np

class EnhancedAdaptiveSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(4, dim * 5)
        self.population_size = self.initial_population_size
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.evaluations = 0
        self.history_best = None
        self.convergence_threshold = 0.01

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(p) for p in population])
        self.evaluations += self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index].copy()

        if self.history_best is None or fitness[best_index] < self.history_best[1]:
            self.history_best = (best_individual.copy(), fitness[best_index])

        mutation_factors = np.random.uniform(0.5, 1.0, self.population_size)

        while self.evaluations < self.budget:
            prev_best_fit = fitness[best_index]

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                trial_mutation_factor = mutation_factors[i]
                trial_vector = np.clip(population[a] + trial_mutation_factor * (population[b] - population[c]),
                                       lower_bound, upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, trial_vector, population[i])

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    mutation_factors[i] = 0.5 * mutation_factors[i] + 0.5 * np.random.uniform(0, 1)

                    if trial_fitness < fitness[best_index]:
                        best_index = i
                        best_individual = trial.copy()
                        if trial_fitness < self.history_best[1]:
                            self.history_best = (best_individual.copy(), trial_fitness)

            if self.evaluations < self.budget:
                worst_index = np.argmax(fitness)
                population[worst_index] = self.history_best[0]
                fitness[worst_index] = func(self.history_best[0])
                self.evaluations += 1

            if abs(prev_best_fit - fitness[best_index]) < self.convergence_threshold:
                self.population_size = min(self.population_size + 1, self.initial_population_size * 2)
                new_individuals = np.random.uniform(lower_bound, upper_bound, (self.population_size - len(population), self.dim))
                population = np.vstack((population, new_individuals))
                new_fitness = np.array([func(p) for p in new_individuals])
                fitness = np.concatenate((fitness, new_fitness))
                mutation_factors = np.concatenate((mutation_factors, np.random.uniform(0.5, 1.0, len(new_individuals))))
                self.evaluations += len(new_individuals)
            else:
                self.population_size = max(self.initial_population_size, self.population_size - 1)
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]
                mutation_factors = mutation_factors[:self.population_size]

        return population[best_index], fitness[best_index]