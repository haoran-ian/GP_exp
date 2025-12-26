import numpy as np

class ImprovedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.98  # Slightly adjusted cooling rate
        self.chaos_coefficient = 0.5
        self.memory_factor = 0.1  # Memory factor for previous best solutions

    def chaotic_map(self, x):
        # Logistic map for chaotic behavior
        return 4 * x * (1 - x)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        best_overall = population[np.argmin(fitness)]  # Track best overall solution

        while eval_count < self.budget:
            chaos_factor = self.chaotic_map(np.random.rand())
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                diff_vector = b - c
                scaling_factor = self.scaling_factor * chaos_factor
                mutant = np.clip(a + scaling_factor * diff_vector + self.memory_factor * (best_overall - a), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate * chaos_factor
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i] or np.exp((fitness[i] - trial_fitness) / self.temperature) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < func(best_overall):  # Update best overall if improved
                        best_overall = trial
            self.temperature *= self.cooling_rate

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]