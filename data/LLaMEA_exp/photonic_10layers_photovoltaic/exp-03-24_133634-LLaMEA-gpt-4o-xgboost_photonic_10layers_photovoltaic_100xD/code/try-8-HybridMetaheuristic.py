import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.dynamic_rate = 0.05  # New dynamic rate for population resizing

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Dynamic population resizing
            if budget_used % (self.budget // 5) == 0:
                self.population_size = max(5, int(self.population_size * (1 - self.dynamic_rate)))
                population = np.vstack((population, np.random.uniform(lb, ub, (self.population_size - len(population), self.dim))))
                fitness = np.append(fitness, [func(ind) for ind in population[len(fitness):]])
                budget_used += self.population_size - len(fitness)

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

            # Clustering for adaptive mutation factor adjustment
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.1 * (ub - lb).mean():
                self.mutation_factor *= 1.1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]