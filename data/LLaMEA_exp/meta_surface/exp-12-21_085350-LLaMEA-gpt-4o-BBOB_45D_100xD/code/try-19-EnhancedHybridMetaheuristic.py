import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.initial_temperature = 100.0
        self.cooling_rate = 0.995
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.cooling_schedule = lambda t: self.initial_temperature * (self.cooling_rate ** t)
        
    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def chaotic_map(self, current_time_step):
        return 0.5 * (1 - np.cos(np.pi * current_time_step))

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        eval_count = self.population_size

        time_step = 0

        while eval_count < self.budget:
            temperature = self.cooling_schedule(time_step)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Dynamic parameter adjustments
                self.inertia_weight = self.chaotic_map(time_step)
                self.cognitive_constant = 1.5 + 0.5 * np.sin(np.pi * time_step / self.budget)
                self.social_constant = 1.5 + 0.5 * np.cos(np.pi * time_step / self.budget)

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
                eval_count += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = fitness

            time_step += 1

        return global_best, global_best_fitness