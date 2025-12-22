import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.99  # Cooling rate for simulated annealing
        self.chaos_coefficient = 0.5  # Initial chaos coefficient

    def chaotic_map(self, x):
        # Logistic map for chaotic behavior
        return 4 * x * (1 - x)

    def levy_flight(self, size):
        # Lévy flight step generation
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return 0.01 * step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            chaos_factor = self.chaotic_map(np.random.rand())
            self.chaos_coefficient = 0.5 * chaos_factor + 0.5 * (self.budget - eval_count) / self.budget
            
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                diff_vector = b - c
                scaling_factor = self.scaling_factor * self.chaos_coefficient
                mutant = np.clip(a + scaling_factor * diff_vector, bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate * self.chaos_coefficient
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Apply Lévy flight for exploration
                trial = trial + self.levy_flight(self.dim)

                # Simulated Annealing Acceptance Criteria with dynamic temperature
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i] or np.exp((fitness[i] - trial_fitness) / self.temperature) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fitness
            
            self.temperature *= self.cooling_rate

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]