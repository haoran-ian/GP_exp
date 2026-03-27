import numpy as np

class HybridDE_SA_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Adjusted cooling rate
        self.F = 0.9  # Adjusted scaling factor
        self.CR = 0.85  # Adjusted crossover probability

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                # Simulated Annealing acceptance with adaptive cooling
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness
                    self.cooling_rate = max(0.9, self.cooling_rate * 1.01)  # Adaptive cooling rate adjustment

            self.temperature *= self.cooling_rate

        return best_solution