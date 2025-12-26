import numpy as np

class EnhancedAdaptiveSwarmDEWithFeedback:
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
        self.learning_rate = 0.1  # New adaptive learning rate

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(p) for p in population])
        self.evaluations += self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index].copy()

        if self.history_best is None or fitness[best_index] < self.history_best[1]:
            self.history_best = (best_individual.copy(), fitness[best_index])

        while self.evaluations < self.budget:
            prev_best_fit = fitness[best_index]

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Multi-strategy mutation
                if np.random.rand() < 0.5:
                    dynamic_mutation_factor = self.mutation_factor * np.random.rand()
                    trial_vector = population[a] + dynamic_mutation_factor * (population[b] - population[c])
                else:
                    trial_vector = population[a] + self.learning_rate * (population[b] - population[c])

                trial_vector = np.clip(trial_vector, lower_bound, upper_bound)

                # Historical best influenced crossover rate
                historical_influence = 0.15 if np.random.rand() < 0.25 else 0
                dynamic_crossover_rate = self.crossover_rate + historical_influence * np.random.rand()
                cross_points = np.random.rand(self.dim) < dynamic_crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, trial_vector, population[i])

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                    if trial_fitness < fitness[best_index]:
                        best_index = i
                        best_individual = trial.copy()
                        if trial_fitness < self.history_best[1]:
                            self.history_best = (best_individual.copy(), trial_fitness)

            # Elite preservation
            if self.evaluations < self.budget:
                worst_index = np.argmax(fitness)
                population[worst_index] = self.history_best[0]
                fitness[worst_index] = func(self.history_best[0])
                self.evaluations += 1

            # Dynamic population size adjustment
            if abs(prev_best_fit - fitness[best_index]) < self.convergence_threshold:
                self.population_size = min(self.population_size + 1, self.initial_population_size * 2)
                new_individuals = np.random.uniform(lower_bound, upper_bound, (self.population_size - len(population), self.dim))
                population = np.vstack((population, new_individuals))
                new_fitness = np.array([func(p) for p in new_individuals])
                fitness = np.concatenate((fitness, new_fitness))
                self.evaluations += len(new_individuals)
            else:
                self.population_size = max(self.initial_population_size, self.population_size - 1)
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]

        return population[best_index], fitness[best_index]