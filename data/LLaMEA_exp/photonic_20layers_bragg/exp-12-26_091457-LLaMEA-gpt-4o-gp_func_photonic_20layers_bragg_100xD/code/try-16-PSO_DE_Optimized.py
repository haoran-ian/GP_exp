import numpy as np

class PSO_DE_Optimized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best = np.copy(self.particles)
        self.global_best = None
        self.fitness = np.full(self.population_size, np.inf)
        self.best_fitness = np.inf
        self.eval_count = 0
        self.w = 0.9  # Dynamic inertia weight for PSO
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.F = 0.8  # Dynamic scaling factor for DE
        self.CR = 0.7  # Crossover probability for DE
        self.prev_best_fitness = np.inf

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                fitness_value = func(self.particles[i])
                self.eval_count += 1

                if fitness_value < self.fitness[i]:
                    self.fitness[i] = fitness_value
                    self.personal_best[i] = self.particles[i]

                if fitness_value < self.best_fitness:
                    self.best_fitness = fitness_value
                    self.global_best = self.particles[i]

            diversity = np.mean(np.std(self.particles, axis=0))
            self._dynamic_update_parameters()

            if diversity > 0.1:
                self._pso_update(bounds)
            else:
                self._de_update(bounds)

        return self.global_best

    def _dynamic_update_parameters(self):
        # Dynamically adjust PSO inertia weight and DE scaling factor
        if self.best_fitness < self.prev_best_fitness:
            self.w = max(0.4, self.w * 0.99)
            self.F = min(0.9, self.F * 1.01)
        else:
            self.w = min(0.9, self.w * 1.02)  # Slightly increased adjustment factor
            self.F = max(0.4, self.F * 0.99)
        self.prev_best_fitness = self.best_fitness

    def _pso_update(self, bounds):
        lr = 0.5  # New adaptive learning rate
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = (lr * self.w * self.velocities[i] +
                                  self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                  self.c2 * r2 * (self.global_best - self.particles[i]))
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], *bounds)

    def _de_update(self, bounds):
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x0, x1, x2 = self.particles[indices]

            mutant_vector = x0 + self.F * (x1 - x2)
            crossover = np.random.rand(self.dim) < self.CR
            trial_vector = np.where(crossover, mutant_vector, self.particles[i])
            trial_vector = np.clip(trial_vector, *bounds)

            if self.eval_count < self.budget:
                trial_fitness = func(trial_vector)
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.particles[i] = trial_vector
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.global_best = trial_vector