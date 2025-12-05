import numpy as np

class AQIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.prob_mutate = 0.1
        self.F = 0.5
        self.CR = 0.9
        self.budget_used = 0

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.array([func(ind) for ind in population])
        self.budget_used += self.population_size

        # Main loop
        while self.budget_used < self.budget:
            new_population = np.copy(population)
            self.population_size = max(5, int(self.population_size * (1 - 0.1 * (self.budget_used / self.budget))))  # Dynamic population
            for i in range(self.population_size):
                if self.budget_used >= self.budget:
                    break

                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F_dynamic = np.random.uniform(0.2, 0.8) * (1 - self.budget_used / self.budget)  # Adaptive F, adjusted range
                mutant = a + F_dynamic * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover = np.random.rand(self.dim) < np.clip(np.random.normal(0.95, 0.02), 0.85, 1.0)  # Modified crossover probability
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.budget_used += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

            # Probabilistic population update
            for i in range(self.population_size):
                if self.budget_used >= self.budget:
                    break
                adaptive_prob_mutate = np.random.uniform(0.1, 0.2) * (1 - self.budget_used / self.budget)  # Adjusted mutation probability
                if np.random.rand() < adaptive_prob_mutate:
                    new_population[i] = self.lower_bound + np.random.rand(self.dim) * (self.upper_bound - self.lower_bound)
                    fitness[i] = func(new_population[i])
                    self.budget_used += 1

            population = new_population

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]