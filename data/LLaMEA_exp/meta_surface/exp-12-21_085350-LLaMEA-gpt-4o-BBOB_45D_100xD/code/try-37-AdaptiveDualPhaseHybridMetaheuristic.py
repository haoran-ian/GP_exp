import numpy as np

class AdaptiveDualPhaseHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate_min = 0.5
        self.crossover_rate_max = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_constant = 1.5
        self.social_constant = 1.5

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
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

        while eval_count < self.budget:
            crossover_rate = self.crossover_rate_min + (self.crossover_rate_max - self.crossover_rate_min) * (eval_count / self.budget)
            inertia_weight = self.inertia_weight_initial - ((self.inertia_weight_initial - self.inertia_weight_final) * (eval_count / self.budget))
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Update velocities using PSO dynamics
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )

                # Crossover and mutation
                if np.random.rand() < crossover_rate:
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    mutant_vector = population[idxs[0]] + self.scaling_factor * (population[idxs[1]] - population[idxs[2]])
                    trial_vector = np.where(np.random.rand(self.dim) < crossover_rate, mutant_vector, population[i])
                    trial_vector = np.clip(trial_vector, bounds[:, 0], bounds[:, 1])
                else:
                    trial_vector = population[i] + velocities[i]

                # Levy flight perturbation
                levy_step = self.levy_flight(0.1 + 0.1 * (eval_count / self.budget))
                if np.random.rand() < 0.2:  # Fixed small mutation probability
                    trial_vector += levy_step

                # Boundary constraints
                trial_vector = np.clip(trial_vector, bounds[:, 0], bounds[:, 1])

                # Evaluate fitness
                fitness = func(trial_vector)
                eval_count += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = trial_vector
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = trial_vector
                        global_best_fitness = fitness

        return global_best, global_best_fitness