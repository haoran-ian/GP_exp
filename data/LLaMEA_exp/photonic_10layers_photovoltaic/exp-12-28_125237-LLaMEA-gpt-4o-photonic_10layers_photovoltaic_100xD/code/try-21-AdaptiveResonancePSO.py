import numpy as np

class AdaptiveResonancePSO:
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
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_initial = 0.9
        self.w_final = 0.4
        self.resonance_factor = 0.1
        self.adaptive_threshold = 0.1

    def initialize_population(self, dim):
        return np.random.rand(self.population_size, dim)

    def __call__(self, func):
        while self.fitness_evaluations < self.budget:
            self.update_particles(func)
            self.apply_resonance(func)
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
            self.particles[i] += self.velocities[i] + self.resonance_factor * np.random.randn(self.dim)
            self.particles[i] = np.clip(self.particles[i], func.bounds.lb, func.bounds.ub)

    def apply_resonance(self, func):
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            perturbation = self.resonance_factor * (np.random.rand(self.dim) - 0.5)
            candidate = self.particles[i] + perturbation
            candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)

            score = func(candidate)
            self.fitness_evaluations += 1

            if score < self.best_particle_scores[i] and np.abs(score - self.best_particle_scores[i]) > self.adaptive_threshold:
                self.particles[i] = candidate
                self.best_particle_scores[i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = candidate