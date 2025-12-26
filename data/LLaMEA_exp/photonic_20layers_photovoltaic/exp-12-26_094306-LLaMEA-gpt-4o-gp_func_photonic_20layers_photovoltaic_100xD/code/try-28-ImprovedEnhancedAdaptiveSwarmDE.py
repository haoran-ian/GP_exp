import numpy as np

class ImprovedEnhancedAdaptiveSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, dim * 5)
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.evaluations = 0
        self.history_best = None
        self.strategy_probabilities = np.array([0.5, 0.5])  # Probabilities for mutation strategies

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

                # Strategy selection based on adaptive probabilities
                strategy = np.random.choice([0, 1], p=self.strategy_probabilities)

                if strategy == 0:  # Classic DE mutation
                    dynamic_mutation_factor = self.mutation_factor * np.random.rand()
                    trial_vector = np.clip(population[a] + dynamic_mutation_factor * (population[b] - population[c]),
                                           lower_bound, upper_bound)
                else:  # Alternative mutation strategy
                    trial_vector = np.clip(population[a] + 0.5 * np.random.rand() * (best_individual - population[b]),
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

                    # Update strategy probabilities based on success
                    self.strategy_probabilities[strategy] += 0.1
                    self.strategy_probabilities = self.strategy_probabilities / np.sum(self.strategy_probabilities)

            # Elite preservation with local search: replace the worst with the historical best if budget allows
            if self.evaluations < self.budget:
                worst_index = np.argmax(fitness)
                population[worst_index] = self.history_best[0]
                fitness[worst_index] = func(self.history_best[0])
                self.evaluations += 1

                # Local search around historical best
                if self.evaluations < self.budget:
                    perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                    local_trial = np.clip(self.history_best[0] + perturbation, lower_bound, upper_bound)
                    local_trial_fitness = func(local_trial)
                    self.evaluations += 1
                    if local_trial_fitness < self.history_best[1]:
                        self.history_best = (local_trial.copy(), local_trial_fitness)
                        best_individual = local_trial.copy()

        return population[best_index], fitness[best_index]