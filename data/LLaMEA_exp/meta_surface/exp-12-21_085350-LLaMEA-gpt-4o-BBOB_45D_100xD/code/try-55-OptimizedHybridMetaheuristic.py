import numpy as np

class OptimizedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 8 * self.dim
        self.final_population_size = 4 * self.dim
        self.crossover_rate = 0.85
        self.scaling_factor = 0.9
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.4
        self.social_constant = 1.6

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population_size = self.initial_population_size
        population = np.random.rand(population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        eval_count = population_size

        while eval_count < self.budget:
            mutation_prob = max(0.05, min(0.25, 1.0 - (eval_count / self.budget)))
            if eval_count % (self.budget // 5) == 0:
                population_size = max(self.final_population_size, population_size - self.dim)

            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
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
                        self.cooling_rate *= 0.99

        return global_best, global_best_fitness