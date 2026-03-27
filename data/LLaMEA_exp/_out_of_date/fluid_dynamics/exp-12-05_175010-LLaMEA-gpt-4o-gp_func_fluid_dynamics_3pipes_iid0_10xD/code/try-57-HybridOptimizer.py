import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim
        self.temp = 1.0
        self.cooling_rate = 0.95

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        best_index = np.argmin(fitness)  # Track best index for elite preservation
        elite_solution = population[best_index].copy()  # Copy the best individual
        elite_fitness = fitness[best_index]

        while self.budget > 0:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutation_factor = 0.3 + 0.3 * np.random.rand()  # Reduced mutation range
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_rate = 0.9
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, population[i])

                trial_fitness = func(trial)
                self.budget -= 1

                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                elif np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / self.temp):  
                    population[i], fitness[i] = trial, trial_fitness

            for i in range(self.population_size):
                perturbation = np.random.normal(0, self.temp, self.dim)
                candidate = np.clip(population[i] + perturbation, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.budget -= 1

                if (candidate_fitness < fitness[i]) or (np.random.rand() < np.exp(-(candidate_fitness - fitness[i]) / self.temp)):
                    population[i], fitness[i] = candidate, candidate_fitness

            self.temp *= self.cooling_rate

            # Preserve the elite solution
            if elite_fitness < np.min(fitness):
                population[np.argmax(fitness)] = elite_solution
                fitness[np.argmax(fitness)] = elite_fitness
            else:
                best_index = np.argmin(fitness)
                elite_solution = population[best_index].copy()
                elite_fitness = fitness[best_index]

            if self.budget < 0.15 * self.budget:  # Adaptive restart condition
                population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                self.budget -= self.population_size

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]