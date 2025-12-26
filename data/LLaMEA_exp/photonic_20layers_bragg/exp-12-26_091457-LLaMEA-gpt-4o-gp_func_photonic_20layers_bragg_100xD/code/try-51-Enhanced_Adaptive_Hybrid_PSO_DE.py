import numpy as np

class Enhanced_Adaptive_Hybrid_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, int(budget / 100))
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best = np.copy(self.particles)
        self.global_best = None
        self.fitness = np.full(self.population_size, np.inf)
        self.best_fitness = np.inf
        self.eval_count = 0
        self.w = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.8
        self.CR = 0.7
        self.prev_best_fitness = np.inf
        self.div_threshold = 0.1
        self.convergence_zone = 0.05  # Convergence zone to switch strategies

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        while self.eval_count < self.budget:
            self._evaluate_fitness(func)
            self._dynamic_update_parameters(bounds)

        return self.global_best

    def _evaluate_fitness(self, func):
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

    def _dynamic_update_parameters(self, bounds):
        diversity = self._calculate_diversity()
        if self.best_fitness < self.prev_best_fitness:
            self.w = max(0.4, self.w * (0.95 + 0.1 * np.random.rand()))
            self.F = min(0.9, self.F * (1.02 + 0.02 * np.random.rand()))
        else:
            self.w = min(0.9, self.w * (1.03 - 0.03 * diversity))
            self.F = max(0.4, self.F * (0.97 + 0.03 * (1 - diversity)))
        self.prev_best_fitness = self.best_fitness

        if self._calculate_convergence() < self.convergence_zone:
            self._selective_pressure_update(bounds)
        else:
            if diversity > self.div_threshold:
                self._pso_update(bounds)
            else:
                self._de_update(bounds)

    def _calculate_diversity(self):
        return np.mean(np.std(self.particles, axis=0))

    def _calculate_convergence(self):
        return np.mean(np.abs(self.global_best - self.particles))

    def _pso_update(self, bounds):
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = (self.w * self.velocities[i] +
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

    def _selective_pressure_update(self, bounds):
        # Introduce selective pressure by emphasizing exploration or exploitation based on the convergence
        for i in range(self.population_size):
            pressure_factor = 1 + np.random.rand() * (1 - self._calculate_convergence())
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = (self.w * self.velocities[i] * pressure_factor +
                                  self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                  self.c2 * r2 * (self.global_best - self.particles[i]))
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], *bounds)