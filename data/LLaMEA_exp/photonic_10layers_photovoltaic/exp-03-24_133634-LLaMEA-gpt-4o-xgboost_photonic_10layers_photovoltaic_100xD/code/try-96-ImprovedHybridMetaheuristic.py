import numpy as np

class ImprovedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Slower cooling for extended exploration
        self.mutation_factor = 0.9
        self.crossover_rate = 0.6
        self.exploration_factor = 0.15  # Increased for dynamic search adaptation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                dynamic_exploration = self.exploration_factor * (np.random.rand() - 0.5)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c) + dynamic_exploration, lb, ub)
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

            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.15 * (ub - lb).mean():
                self.mutation_factor *= 1.1
                self.exploration_factor *= 1.2

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]