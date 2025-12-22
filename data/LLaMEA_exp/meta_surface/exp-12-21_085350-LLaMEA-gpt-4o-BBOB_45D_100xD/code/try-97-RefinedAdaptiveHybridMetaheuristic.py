import numpy as np

class RefinedAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.inertia_weight = 0.9  # Increased inertia weight for better exploration
        self.cognitive_constant = 1.5
        self.social_constant = 1.7
        self.chaotic_map_factor = 0.5  # Chaotic map factor for dynamic parameter adjustments
        self.sub_population_size = max(2, self.population_size // 3)  # Split into smaller interacting sub-populations

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def chaotic_map(self, x):
        return np.sin(self.chaotic_map_factor * np.pi * x)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        eval_count = self.population_size

        while eval_count < self.budget:
            mutation_prob = max(0.1, min(0.3, 1.0 - (eval_count / self.budget)))
            chaotic_inertia = self.chaotic_map(eval_count / self.budget)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                if i % self.sub_population_size == 0 and i != 0:
                    sub_population_best = personal_best[np.argmin(personal_best_fitness[i - self.sub_population_size:i])]
                    if func(sub_population_best) < global_best_fitness:
                        global_best = sub_population_best
                        global_best_fitness = func(sub_population_best)

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    (self.inertia_weight * chaotic_inertia) * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )

                population[i] += velocities[i]

                levy_step = self.levy_flight(0.1 + 0.1 * (eval_count / self.budget))
                if np.random.rand() < mutation_prob:
                    population[i] += levy_step

                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

                fitness = func(population[i])
                eval_count += 1

                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = fitness
                        self.cooling_rate *= 0.993

        return global_best, global_best_fitness