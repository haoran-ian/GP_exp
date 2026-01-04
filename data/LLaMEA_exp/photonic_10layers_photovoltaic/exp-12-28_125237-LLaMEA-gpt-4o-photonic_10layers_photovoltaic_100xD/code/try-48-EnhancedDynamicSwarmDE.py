import numpy as np

class EnhancedDynamicSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = self.initialize_population(dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.best_particle_positions = np.copy(self.particles)
        self.best_particle_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.fitness_evaluations = 0
        self.c1 = 1.5
        self.c2 = 2.5
        self.w_initial = 0.8
        self.w_final = 0.3
        self.turbulence_factor = 0.03
        self.f_min = 0.3
        self.f_max = 0.8
        self.cr = 0.8

    def initialize_population(self, dim):
        x = 0.7
        seq = np.empty(self.population_size)
        for i in range(self.population_size):
            x = 4.0 * x * (1.0 - x)
            seq[i] = x
        return np.random.rand(self.population_size, dim) * seq[:, np.newaxis]

    def levy_flight(self, lam=1.5):
        sigma_u = np.power((np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2)) /
                           (np.math.gamma((1 + lam) / 2) * lam * np.power(2, ((lam - 1) / 2))), 1 / lam)
        u = np.random.normal(0, sigma_u, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.power(np.abs(v), 1 / lam)
        return step

    def __call__(self, func):
        while self.fitness_evaluations < self.budget:
            self.update_particles(func)
            self.apply_adaptive_differential_evolution(func)
            if np.random.rand() < 0.3:
                self.apply_levy_perturbation(func)
        return self.global_best_position

    def update_particles(self, func):
        w = self.w_initial - (self.w_initial - self.w_final) * (self.fitness_evaluations / self.budget)
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            score = func(self.particles[i])
            self.fitness_evaluations += 1

            if score < self.best_particle_scores[i]:
                self.best_particle_scores[i] = score
                self.best_particle_positions[i] = self.particles[i].copy()

            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i].copy()

        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            velocity_cognitive = self.c1 * r1 * (self.best_particle_positions[i] - self.particles[i])
            velocity_social = self.c2 * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = w * self.velocities[i] + velocity_cognitive + velocity_social
            self.particles[i] += self.velocities[i] + self.turbulence_factor * np.random.randn(self.dim)
            self.particles[i] = np.clip(self.particles[i], func.bounds.lb, func.bounds.ub)

    def apply_adaptive_differential_evolution(self, func):
        diversity = np.mean(np.std(self.particles, axis=0))
        f = self.f_min + (self.f_max - self.f_min) * (diversity / (np.max([diversity, 1e-6])))
        
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.particles[a] + f * (self.particles[b] - self.particles[c])
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

            if self.global_best_score < np.median(self.best_particle_scores):
                self.cr = 0.55  # Adjusted line for crossover rate
                self.turbulence_factor = 0.01
            else:
                self.cr = 0.95
                self.turbulence_factor = 0.1

            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, self.particles[i])

            score = func(trial)
            self.fitness_evaluations += 1

            if score < self.best_particle_scores[i]:
                self.particles[i] = trial
                self.best_particle_scores[i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = trial

    def apply_levy_perturbation(self, func):
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            levy_step = self.levy_flight()
            candidate = self.particles[i] + levy_step
            candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)

            score = func(candidate)
            self.fitness_evaluations += 1

            if score < self.best_particle_scores[i]:
                self.particles[i] = candidate
                self.best_particle_scores[i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = candidate