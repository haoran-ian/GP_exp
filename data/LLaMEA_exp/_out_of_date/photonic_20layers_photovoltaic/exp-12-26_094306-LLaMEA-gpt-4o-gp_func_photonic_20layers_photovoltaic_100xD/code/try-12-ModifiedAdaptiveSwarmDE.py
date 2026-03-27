import numpy as np

class ModifiedAdaptiveSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, dim * 5)
        self.crossover_rate = 0.9
        self.initial_mutation_factor = 0.8
        self.evaluations = 0

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index].copy()

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = np.array([j for j in range(self.population_size) if j != i])
                chosen_indices = np.random.choice(indices, 3, replace=False)
                a, b, c = chosen_indices

                # Adjust mutation factor dynamically
                dynamic_mutation_factor = self.initial_mutation_factor * (0.5 + 0.5 * np.random.rand()) * (1 - (self.evaluations / self.budget))
                
                # Group-based dynamic mutation strategy
                group_center = (population[a] + population[b] + population[c]) / 3
                trial_vector = np.clip(group_center + dynamic_mutation_factor * (population[b] - population[c]), 
                                       lower_bound, upper_bound)

                # Adjust crossover rate dynamically
                dynamic_crossover_rate = self.crossover_rate + 0.2 * (np.random.rand() - 0.5)
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

            if self.evaluations < self.budget:
                worst_index = np.argmax(fitness)
                if fitness[worst_index] > fitness[best_index]:
                    population[worst_index] = best_individual
                    fitness[worst_index] = func(best_individual)
                    self.evaluations += 1

        return population[best_index], fitness[best_index]