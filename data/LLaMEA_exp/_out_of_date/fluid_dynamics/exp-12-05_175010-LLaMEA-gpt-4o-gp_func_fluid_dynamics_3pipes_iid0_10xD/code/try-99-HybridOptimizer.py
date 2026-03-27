import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim
        self.temp = 1.0
        self.cooling_rate = 0.9  # Adjusted cooling rate
        self.initial_budget = budget

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        while self.budget > 0:
            best_index = np.argmin(fitness)
            best_individual = population[best_index]

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutation_factor = 0.2 + 0.8 * (self.budget / float(self.initial_budget))  # Dynamic mutation factor
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_rate = 0.9  # Adjusted crossover rate
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

            if self.budget < 0.2 * self.initial_budget:  # Adjusted adaptive restart condition
                population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                self.budget -= self.population_size

            if fitness[best_index] < np.min(fitness):
                population[np.argmax(fitness)] = best_individual

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]