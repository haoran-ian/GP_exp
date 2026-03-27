import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 15 * dim
        self.initial_temperature = 1.0
        self.cooling_rate = 0.98 + np.random.rand() * 0.02
        self.F = 0.5 + np.random.rand() * 0.5
        self.CR = 0.9 + np.random.rand() * 0.1  # Adaptive crossover rate

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        temperature = self.initial_temperature

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(a + self.F * (population[b] - population[c] + (best_solution - population[i])), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness
                elif np.random.rand() < 0.05:  # Random reinitialization strategy
                    population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    fitness[i] = func(population[i])
                    evaluations += 1

            temperature *= self.cooling_rate

        return best_solution