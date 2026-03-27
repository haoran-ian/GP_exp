import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.mutation_factor = 0.9
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Enhanced Differential Evolution: mutate and crossover
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Adaptive Simulated Annealing: accept based on dynamic acceptance probability
                trial_fitness = func(trial)
                budget_used += 1
                acceptance_probability = np.exp((fitness[i] - trial_fitness) / self.temperature)
                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break
            
            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Evolving Neighborhood Search for local refinement
            if np.random.rand() < 0.3:  # Probability to intensify search in neighborhood
                best_idx = np.argmin(fitness)
                neighborhood = population[best_idx] + np.random.normal(0, 0.05, size=(10, self.dim))
                neighborhood = np.clip(neighborhood, lb, ub)
                neighborhood_fitness = np.array([func(ind) for ind in neighborhood])
                budget_used += len(neighborhood)
                if neighborhood_fitness.min() < fitness[best_idx]:
                    best_neighborhood_idx = np.argmin(neighborhood_fitness)
                    population[best_idx] = neighborhood[best_neighborhood_idx]
                    fitness[best_idx] = neighborhood_fitness[best_neighborhood_idx]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]