import numpy as np

class DynamicAdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.995
        self.inertia_weight = 0.5
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.elite_fraction = 0.1
        self.dynamic_population_growth_rate = 1.01

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
            mutation_prob = max(0.1, min(0.3, 1.0 - (eval_count / self.budget)))
            population_size = int(min(self.initial_population_size * self.dynamic_population_growth_rate, self.budget - eval_count))
            elite_size = max(1, int(self.elite_fraction * population_size))
            elite_indices = np.argsort(personal_best_fitness)[:elite_size]

            new_population = np.copy(personal_best[elite_indices])
            new_velocities = np.copy(velocities[elite_indices])

            for i in range(elite_size, population_size):
                if eval_count >= self.budget:
                    break
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia = self.inertia_weight * new_velocities[np.random.choice(elite_size)]
                cognitive = self.cognitive_constant * r1 * (personal_best[i % elite_size] - new_population[i % elite_size])
                social = self.social_constant * r2 * (global_best - new_population[i % elite_size])
                new_velocity = inertia + cognitive + social
                new_position = new_population[i % elite_size] + new_velocity

                levy_step = self.levy_flight(0.1 + 0.1 * (eval_count / self.budget))
                if np.random.rand() < mutation_prob:
                    new_position += levy_step

                new_position = np.clip(new_position, bounds[:, 0], bounds[:, 1])
                fitness = func(new_position)
                eval_count += 1

                if fitness < personal_best_fitness[i % elite_size]:
                    personal_best[i % elite_size] = new_position
                    personal_best_fitness[i % elite_size] = fitness
                    if fitness < global_best_fitness:
                        global_best = new_position
                        global_best_fitness = fitness

                new_population = np.vstack((new_population, new_position))
                new_velocities = np.vstack((new_velocities, new_velocity))

            population = new_population[:population_size]
            velocities = new_velocities[:population_size]

        return global_best, global_best_fitness