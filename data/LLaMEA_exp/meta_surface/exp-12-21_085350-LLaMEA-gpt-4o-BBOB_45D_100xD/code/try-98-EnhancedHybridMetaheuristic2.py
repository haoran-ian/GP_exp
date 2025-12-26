import numpy as np

class EnhancedHybridMetaheuristic2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * self.dim
        self.min_population_size = 5 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.7

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
            adaptive_inertia = self.inertia_weight * (1 - eval_count / self.budget)
            adaptive_cognitive = self.cognitive_constant * (1 + eval_count / self.budget)
            adaptive_social = self.social_constant * (1 - eval_count / self.budget)
            
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    adaptive_inertia * velocities[i]
                    + adaptive_cognitive * r1 * (personal_best[i] - population[i])
                    + adaptive_social * r2 * (global_best - population[i])
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

            # Dynamic population scaling
            if eval_count % (self.budget // 10) == 0:
                population_size = max(
                    self.min_population_size,
                    int(self.initial_population_size * (1 - eval_count / self.budget))
                )
                population = population[:population_size]
                velocities = velocities[:population_size]
                personal_best = personal_best[:population_size]
                personal_best_fitness = personal_best_fitness[:population_size]
                
        return global_best, global_best_fitness