import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim)
        self.personal_best = np.copy(self.particles)
        self.global_best = None
        self.fitness = np.full(self.population_size, np.inf)
        self.best_fitness = np.inf
        self.eval_count = 0
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.F = 0.5  # Scaling factor for DE
        self.CR = 0.7  # Crossover probability for DE

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                # Evaluate fitness of particles
                if self.eval_count >= self.budget:
                    break
                fitness_value = func(self.particles[i])
                self.eval_count += 1

                # Update personal best
                if fitness_value < self.fitness[i]:
                    self.fitness[i] = fitness_value
                    self.personal_best[i] = self.particles[i]

                # Update global best
                if fitness_value < self.best_fitness:
                    self.best_fitness = fitness_value
                    self.global_best = self.particles[i]

            # Calculate diversity
            diversity = np.mean(np.std(self.particles, axis=0))

            # Switch between PSO and DE according to the diversity
            if diversity > 0.1:
                self._pso_update(bounds)
            else:
                self._de_update(bounds)

        return self.global_best

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

            # Evaluate the trial vector
            if self.eval_count < self.budget:
                trial_fitness = func(trial_vector)
                self.eval_count += 1

                # Replace if the trial vector is better
                if trial_fitness < self.fitness[i]:
                    self.particles[i] = trial_vector
                    self.fitness[i] = trial_fitness

                    # Update personal and global bests
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.global_best = trial_vector