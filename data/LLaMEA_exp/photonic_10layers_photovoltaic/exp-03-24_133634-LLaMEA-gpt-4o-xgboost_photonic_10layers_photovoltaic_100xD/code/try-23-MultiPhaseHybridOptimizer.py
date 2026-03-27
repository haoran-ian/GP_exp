import numpy as np

class MultiPhaseHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.92  # Further refined cooling rate
        self.mutation_factor = 0.9  # Increased mutation factor for exploration
        self.crossover_rate = 0.75  # Slightly increased crossover rate
        self.niching_radius = 0.1  # Initial niching radius

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        while budget_used < self.budget:
            # Adaptive Differential Evolution with Niching
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.sin(3 * budget_used / self.budget * np.pi)  # More dynamic
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Enhanced Simulated Annealing with dynamic temperature
                trial_fitness = func(trial)
                budget_used += 1
                acceptance_prob = np.exp((fitness[i] - trial_fitness) / (self.temperature * (1 + np.linalg.norm(population[i] - trial) / self.niching_radius)))
                if trial_fitness < fitness[i] or np.random.rand() < acceptance_prob:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if budget_used >= self.budget:
                    break
            
            # Cool down temperature and adapt niching
            self.temperature *= self.cooling_rate
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.15 * (ub - lb).mean():
                self.mutation_factor *= 1.2  # Enhanced adaptive factor
                self.niching_radius *= 0.9  # Reduce niching radius to focus search

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]