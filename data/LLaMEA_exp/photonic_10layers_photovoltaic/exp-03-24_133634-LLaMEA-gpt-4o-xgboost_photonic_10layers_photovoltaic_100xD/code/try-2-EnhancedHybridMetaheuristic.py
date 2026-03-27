import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.initial_temperature = 1.0
        self.cooling_rate = 0.95
        self.mutation_factor_min = 0.5
        self.mutation_factor_max = 0.9
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        temperature = self.initial_temperature

        def chaotic_sequence(x, beta=2):
            return beta * x * (1 - x)

        chaotic_initial = np.random.rand()
        chaotic_value = chaotic_sequence(chaotic_initial)

        while budget_used < self.budget:
            self.mutation_factor = self.mutation_factor_min + chaotic_value * (self.mutation_factor_max - self.mutation_factor_min)
            
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break
            
            temperature *= self.cooling_rate
            chaotic_value = chaotic_sequence(chaotic_value)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]