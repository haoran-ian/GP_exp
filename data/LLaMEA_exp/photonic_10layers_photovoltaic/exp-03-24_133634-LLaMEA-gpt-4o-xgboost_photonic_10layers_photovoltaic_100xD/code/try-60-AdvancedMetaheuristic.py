import numpy as np

class AdvancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.initial_temperature = 1.0
        self.cooling_rate = 0.9
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.temperature = self.initial_temperature

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size
        
        def adjust_mutation_factor(diversity):
            if diversity > 0.2:
                return max(0.5, self.mutation_factor / 1.1)
            elif diversity < 0.1:
                return min(1.2, self.mutation_factor * 1.1)
            return self.mutation_factor
        
        while budget_used < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = adjust_mutation_factor(np.std(population, axis=0).mean())
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.1 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if budget_used >= self.budget:
                    break

            self.temperature *= self.cooling_rate
            self.temperature = max(self.temperature, 0.01)
            
            diversity = np.std(population, axis=0).mean()
            self.mutation_factor = adjust_mutation_factor(diversity)
            
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]