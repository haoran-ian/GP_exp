import numpy as np

class ImprovedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.98  # Slightly increased cooling for more exploration
        self.chaos_coefficient = 0.5

    def chaotic_map(self, x):
        return 4 * x * (1 - x)

    def quantum_init(self, bounds):
        # Quantum-inspired initialization to enhance diversity
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim)) + \
               np.random.randn(self.population_size, self.dim) * 0.1

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = self.quantum_init(bounds)
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            chaos_factor = self.chaotic_map(np.random.rand())
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                diff_vector = b - c
                scaling_factor = self.scaling_factor * chaos_factor
                mutant = np.clip(a + scaling_factor * diff_vector, bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i] or np.exp((fitness[i] - trial_fitness) / self.temperature) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fitness
            
            self.temperature *= self.cooling_rate
            
            # Periodic contraction of search space for intensified search
            if eval_count % (self.budget // 10) == 0:
                contraction_factor = 0.9
                bounds_center = (bounds[:, 0] + bounds[:, 1]) / 2
                bounds_range = (bounds[:, 1] - bounds[:, 0]) * contraction_factor
                bounds = np.array([bounds_center - bounds_range/2, bounds_center + bounds_range/2]).T

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]