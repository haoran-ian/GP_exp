import numpy as np

class AdvancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Further adjusted cooling rate
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Dynamic Clustering for diversity control
            cluster_centers = np.unique(population // ((ub - lb) / 10), axis=0)
            cluster_sizes = np.array([np.sum((population // ((ub - lb) / 10) == center).all(axis=1)) for center in cluster_centers])
            entropy = -np.sum((cluster_sizes / self.population_size) * np.log(cluster_sizes / self.population_size + 1e-9))
            if entropy < np.log(self.population_size) * 0.7:
                self.mutation_factor *= 1.2

            # Differential Evolution: mutate and crossover
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Gradient-based local search
                gradient_step = 0.01 * (ub - lb) / self.dim
                gradient = np.array([func(np.clip(trial + epsilon * np.eye(1, self.dim, j)[0], lb, ub)) - trial_fitness for j, epsilon in enumerate(gradient_step)])
                trial = np.clip(trial - gradient_step * gradient, lb, ub)

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