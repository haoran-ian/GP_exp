import numpy as np

class ImprovedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.inertia_weight = 0.5
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.adaptive_weight_factor = 0.05  # New factor for adaptive weight adjustment

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def stochastic_perturbation(self, individual):
        perturbation = np.random.normal(0, 1, self.dim) * 0.1  # Small stochastic perturbation
        return individual + perturbation

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
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Update velocities using PSO dynamics with adaptive inertia weight
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.inertia_weight = 0.9 - (0.9 - 0.4) * (eval_count / self.budget)
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

                # Stochastic perturbation
                if np.random.rand() < 0.1:
                    population[i] = self.stochastic_perturbation(population[i])

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
                        self.cooling_rate *= 0.995  # Adaptive cooling rate

        return global_best, global_best_fitness