import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Faster cooling for more exploitation
        self.mutation_factor = 0.8
        self.crossover_rate = 0.75  # Slightly increased crossover rate
        self.population_size = self.initial_population_size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Dynamic population adjustment
            if np.random.rand() < 0.1:
                self.population_size = max(5 * self.dim, int(self.population_size * 0.9))
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]

            # Differential Evolution: mutate and crossover
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Simulated Annealing: accept based on Metropolis criterion
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break
            
            # Cool down temperature
            self.temperature *= self.cooling_rate
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]