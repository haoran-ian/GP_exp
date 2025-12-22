import numpy as np

class AdaptiveMultiStrategyMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.initial_temperature = 100.0
        self.cooling_rate = 0.995
        self.inertia_weight = 0.7
        self.cognitive_constant = 2.0
        self.social_constant = 2.0

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

        current_temperature = self.initial_temperature

        while eval_count < self.budget:
            mutation_prob = max(0.05, min(0.25, 1.0 - (eval_count / self.budget)))  # Adaptive mutation probability
            
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Update velocities using PSO dynamics
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )

                # Differential Evolution (DE) inspired mutation
                candidates = np.random.choice(self.population_size, 3, replace=False)
                mutant_vector = population[candidates[0]] + self.scaling_factor * (population[candidates[1]] - population[candidates[2]])
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                population[i] = np.clip(trial_vector, bounds[:, 0], bounds[:, 1])

                # Simulated Annealing inspired acceptance
                trial_fitness = func(population[i])
                eval_count += 1
                delta_fitness = trial_fitness - personal_best_fitness[i]
                if delta_fitness < 0 or np.exp(-delta_fitness / current_temperature) > np.random.rand():
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = trial_fitness
                    if trial_fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = trial_fitness
                        self.cooling_rate *= 0.995  # Adaptive cooling rate

                # Levy flight perturbation
                levy_step = self.levy_flight(0.1 + 0.1 * (eval_count / self.budget))  # Dynamic search space adjustment
                if np.random.rand() < mutation_prob:
                    population[i] += levy_step
                    population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

            current_temperature *= self.cooling_rate  # Reduce temperature

        return global_best, global_best_fitness