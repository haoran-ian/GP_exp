import numpy as np

class EnhancedAdaptiveSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, dim * 5)
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.evaluations = 0
        self.history_best = None

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
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                dynamic_mutation_factor = self.mutation_factor * np.random.rand()
                trial_vector = np.clip(population[a] + dynamic_mutation_factor * (population[b] - population[c]) + 0.1 * np.mean([population[a], population[b], population[c]], axis=0),
                                       lower_bound, upper_bound)

                # Historical best influenced crossover rate
                historical_influence = 0.1 if np.random.rand() < 0.2 else 0
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

            # Elite preservation: replace the worst individual with the historical best if the budget allows
            if self.evaluations < self.budget:
                worst_index = np.argmax(fitness)
                population[worst_index] = self.history_best[0]
                fitness[worst_index] = func(self.history_best[0])
                self.evaluations += 1

        return population[best_index], fitness[best_index]