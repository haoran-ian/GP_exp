import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, int(8 * dim))  # Changed population size formula
        self.initial_temperature = 1.0
        self.cooling_rate = 0.93  # Changed for even slower cooling
        self.F = 0.8
        self.CR = 0.9

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
            new_population_size = max(5, int(self.population_size * (1 - evaluations / self.budget)))  # Adaptive pop size
            population = population[:new_population_size]
            fitness = fitness[:new_population_size]

            for i in range(new_population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution
                idxs = [idx for idx in range(new_population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(population[a] + self.F * (population[b] - population[c]), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                # Simulated Annealing acceptance
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            temperature *= self.cooling_rate

        return best_solution