import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.93
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.successful_mutations = 0
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            success_ratio = self.successful_mutations / max(1, budget_used)
            self.mutation_factor = 0.8 * (1 + 0.5 * success_ratio)

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                budget_used += 1
                
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness
                    self.successful_mutations += 1

                if budget_used >= self.budget:
                    break
            
            self.temperature *= self.cooling_rate
            
            current_best_fitness = np.min(fitness)
            if budget_used % (self.population_size * 5) == 0:  # Introduce diversity periodically
                diversity_threshold = 0.1 * (ub - lb).mean()
                diversity = np.linalg.norm(population - np.mean(population, axis=0), axis=1).mean()
                if diversity < diversity_threshold:
                    indices = np.random.choice(self.population_size, size=self.dim, replace=False)
                    population[indices] = np.random.uniform(lb, ub, (len(indices), self.dim))
                    fitness[indices] = [func(ind) for ind in population[indices]]
                    budget_used += len(indices)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]