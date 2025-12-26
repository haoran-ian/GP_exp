import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.7
        self.velocity_clamp_factor = 0.5  # New velocity clamping factor

    def levy_flight(self, L, scale_factor):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = scale_factor * u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        eval_count = self.population_size

        velocity_clamp = (bounds[:, 1] - bounds[:, 0]) * self.velocity_clamp_factor
        
        while eval_count < self.budget:
            mutation_prob = max(0.1, min(0.3, 1.0 - (eval_count / self.budget)))
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )
                
                # Apply velocity clamping
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)

                population[i] += velocities[i]

                dynamic_levy_scale = 0.1 + 0.2 * (eval_count / self.budget)
                levy_step = self.levy_flight(dynamic_levy_scale, scale_factor=0.5)

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