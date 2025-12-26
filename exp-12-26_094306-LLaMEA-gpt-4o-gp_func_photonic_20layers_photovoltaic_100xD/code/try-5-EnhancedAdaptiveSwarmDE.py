import numpy as np

class EnhancedAdaptiveSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, dim * 5)
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.evaluations = 0

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(p) for p in population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                dynamic_mutation_factor = self.mutation_factor * (1 - (self.evaluations / self.budget))
                trial_vector = np.clip(population[a] + dynamic_mutation_factor * (population[b] - population[c]), 
                                       lower_bound, upper_bound)

                dynamic_crossover_rate = self.crossover_rate + 0.1 * np.random.rand()
                cross_points = np.random.rand(self.dim) < dynamic_crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, trial_vector, population[i])

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

            # Introduce adaptive population scaling and elite preservation
            elite_idx = np.argmin(fitness)
            population = np.append(population, population[elite_idx:elite_idx+1], axis=0)
            fitness = np.append(fitness, fitness[elite_idx])

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]