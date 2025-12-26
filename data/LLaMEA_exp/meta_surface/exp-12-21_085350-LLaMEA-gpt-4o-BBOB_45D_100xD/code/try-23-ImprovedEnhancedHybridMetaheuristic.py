import numpy as np

class ImprovedEnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.inertia_weight = 0.9  # Increased for better exploration
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.eval_count = 0

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def dynamic_parameter_tuning(self):
        # Dynamically adjust parameters based on progress
        progress = self.eval_count / self.budget
        self.inertia_weight = max(0.4, 0.9 - 0.5 * progress)  # Decrease inertia over time
        self.cooling_rate = 0.99 - 0.04 * progress  # Adjust cooling rate

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        self.eval_count = self.population_size

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                # Update velocities using PSO dynamics
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )

                # Update position
                population[i] += velocities[i]

                # Levy flight perturbation
                levy_step = self.levy_flight(0.1)
                if np.random.rand() < 0.3:
                    population[i] += levy_step

                # Boundary constraints
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

                # Evaluate fitness
                fitness = func(population[i])
                self.eval_count += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = fitness

            # Dynamic parameter tuning
            self.dynamic_parameter_tuning()

        return global_best, global_best_fitness