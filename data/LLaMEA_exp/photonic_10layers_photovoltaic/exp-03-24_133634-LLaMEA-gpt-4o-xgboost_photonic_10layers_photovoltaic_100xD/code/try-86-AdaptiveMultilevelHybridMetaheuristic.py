import numpy as np

class AdaptiveMultilevelHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Slower cooling to maintain exploration
        self.mutation_factor = 0.9
        self.crossover_rate = 0.7
        self.exploration_factor = 0.1
        self.restart_threshold = 0.05  # Threshold for restart based on diversity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Differential Evolution with multilevel exploration
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                dynamic_exploration = self.exploration_factor * (np.random.rand() - 0.5)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c) + dynamic_exploration, lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
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

            # Dynamic adjustment for mutation and exploration factors with clustering
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.1 * (ub - lb).mean():
                self.mutation_factor *= 1.2
                self.exploration_factor *= 1.1
            else:
                self.mutation_factor *= 0.9  # Reduce if diversity is high
                self.exploration_factor *= 0.9

            # Restart mechanism based on diversity
            if diversity < self.restart_threshold * (ub - lb).mean():
                population = np.random.uniform(lb, ub, (self.population_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                budget_used += self.population_size

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]