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
        self.inertia_weight = 0.5
        self.cognitive_constant = 1.5
        self.social_constant = 1.5

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def differential_evolution(self, population, bounds, mutation_prob):
        for i in range(self.population_size):
            if np.random.rand() < mutation_prob:
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.scaling_factor * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                if np.random.rand() < mutation_prob:
                    trial += self.levy_flight(0.1)
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])
                yield trial

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
            for i, trial in enumerate(self.differential_evolution(population, bounds, mutation_prob)):
                if eval_count >= self.budget:
                    break

                # Evaluate trial solution
                fitness = func(trial)
                eval_count += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = trial
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = trial
                        global_best_fitness = fitness
                        self.cooling_rate *= 0.995

            # Update velocities and positions using PSO dynamics
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], bounds[:, 0], bounds[:, 1])

        return global_best, global_best_fitness