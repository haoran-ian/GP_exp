import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim  # Reduced Population size for Differential Evolution
        self.temp = 1.0  # Initial temperature for Simulated Annealing
        self.cooling_rate = 0.95  # Accelerated cooling rate for Simulated Annealing
        self.mutation_factor = 0.8  # Initial mutation factor
        self.crossover_prob = 0.9  # Initial crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        # Main optimization loop
        while self.budget > 0:
            for i in range(self.population_size):
                # Differential Evolution-like mutation and crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant, population[i])

                # Evaluate new candidate solution
                trial_fitness = func(trial)
                self.budget -= 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness

            # Adapt mutation factor and crossover probability
            self.mutation_factor = 0.5 + 0.3 * np.random.rand()
            self.crossover_prob = 0.7 + 0.2 * np.random.rand()

            # Simulated Annealing-like perturbation
            for i in range(self.population_size):
                perturbation = np.random.normal(0, self.temp, self.dim)
                candidate = np.clip(population[i] + perturbation, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                self.budget -= 1

                # Metropolis acceptance criterion
                if (candidate_fitness < fitness[i]) or (np.random.rand() < np.exp(-(candidate_fitness - fitness[i]) / self.temp)):
                    population[i], fitness[i] = candidate, candidate_fitness

            # Cool down temperature
            self.temp *= self.cooling_rate

        # Return the best solution found
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]