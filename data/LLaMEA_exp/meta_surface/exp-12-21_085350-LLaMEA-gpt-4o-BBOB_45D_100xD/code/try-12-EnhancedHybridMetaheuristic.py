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
        self.inertia_weight = 0.7  # Adjusted inertia weight for better balance
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.mutation_factor = 0.5  # Added mutation factor for differential evolution
        self.adaptive_rate = 0.05  # Adaptive rate for parameter tuning

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def differential_evolution(self, population, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, func.bounds.lb, func.bounds.ub)

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

                # Update velocities using PSO dynamics
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )

                # Adaptive parameter tuning
                if np.random.rand() < self.adaptive_rate:
                    self.inertia_weight = np.random.uniform(0.5, 0.9)
                    self.cognitive_constant = np.random.uniform(1.0, 2.0)
                    self.social_constant = np.random.uniform(1.0, 2.0)
                    self.mutation_factor = np.random.uniform(0.3, 0.7)

                # Differential evolution crossover
                donor = self.differential_evolution(population, i)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, donor, population[i])

                # Levy flight perturbation
                levy_step = self.levy_flight(0.1)
                if np.random.rand() < 0.3:
                    trial += levy_step

                # Boundary constraints
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                # Evaluate fitness
                fitness = func(trial)
                eval_count += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = trial
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = trial
                        global_best_fitness = fitness

        return global_best, global_best_fitness