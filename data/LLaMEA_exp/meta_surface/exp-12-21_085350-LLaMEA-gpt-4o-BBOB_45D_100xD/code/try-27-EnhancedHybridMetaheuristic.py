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
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_constant = 1.5
        self.social_constant = 1.5

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def chaotic_initialization(self, size, bounds):
        return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.sin(np.pi * np.random.rand(size, self.dim))

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = self.chaotic_initialization(self.population_size, bounds)
        velocities = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        eval_count = self.population_size
        
        while eval_count < self.budget:
            inertia_weight = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (eval_count / self.budget)
            social_influence = self.social_constant * (1 + np.cos(np.pi * eval_count / self.budget))
            mutation_prob = max(0.1, min(0.3, 1.0 - (eval_count / self.budget)))
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Update velocities using PSO dynamics
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + social_influence * r2 * (global_best - population[i])
                )

                # Update position
                population[i] += velocities[i]

                # Levy flight perturbation
                levy_step = self.levy_flight(0.1 + 0.1 * (eval_count / self.budget))
                if np.random.rand() < mutation_prob:
                    population[i] += levy_step

                # Boundary constraints
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

                # Evaluate fitness
                fitness = func(population[i])
                eval_count += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = fitness

        return global_best, global_best_fitness