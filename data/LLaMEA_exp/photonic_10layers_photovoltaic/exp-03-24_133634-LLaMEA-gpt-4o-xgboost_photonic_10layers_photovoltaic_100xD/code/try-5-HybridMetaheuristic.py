import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        
        chaotic_sequence = np.sin(np.arange(1, self.budget+1))  # Chaotic sequence for exploration

        while budget_used < self.budget:
            factor = 1 + chaotic_sequence[budget_used % self.budget] * 0.1  # Dynamic scaling factor
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c) * factor, lb, ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break
            
            self.temperature *= self.cooling_rate
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]