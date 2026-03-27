import numpy as np

class AdvancedSelfAdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Slower cooling for sustained exploration
        self.base_mutation_factor = 0.8
        self.base_crossover_rate = 0.7
        self.mutation_boost = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Adaptive mutation and crossover based on diversity
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.1 * (ub - lb).mean():
                mutation_factor = self.base_mutation_factor * (1 + self.mutation_boost)
            else:
                mutation_factor = self.base_mutation_factor
            
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                dynamic_mutation = mutation_factor * (b - c)
                mutant = np.clip(a + dynamic_mutation, lb, ub)
                crossover_rate = self.base_crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Simulated Annealing: Metropolis acceptance criterion
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