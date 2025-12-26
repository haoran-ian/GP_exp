import numpy as np

class AdvancedHybridMetaheuristic:
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
        self.social_constant = 1.9  # Enhanced social component for exploration

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def stochastic_tunneling(self, fitness):
        return 1 - np.exp(-self.temperature * (fitness - np.min(fitness)))

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
            mutation_prob = max(0.1, min(0.3, 1.0 - (eval_count / self.budget)))
            
            # Adaptive population resizing
            reduced_population_size = int(self.population_size * (1 - (eval_count / self.budget) * 0.5))
            if reduced_population_size < 2:
                reduced_population_size = 2

            for i in range(reduced_population_size):
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

                # Update position
                population[i] += velocities[i]

                # Levy flight perturbation
                levy_step = self.levy_flight(0.1 + 0.1 * (eval_count / self.budget))
                if np.random.rand() < mutation_prob:
                    population[i] += levy_step

                # Boundary constraints
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

                # Evaluate fitness with stochastic tunneling
                fitness = func(population[i])
                eval_count += 1
                fitness_adjusted = self.stochastic_tunneling(fitness)

                # Update personal and global bests
                if fitness_adjusted < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness_adjusted
                    if fitness_adjusted < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = fitness_adjusted
                        self.cooling_rate *= 0.993

        return global_best, global_best_fitness