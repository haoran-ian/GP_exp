import numpy as np

class AdaptiveMultiPhaseSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.sub_populations = 3
        self.particles = [self.initialize_population(dim) for _ in range(self.sub_populations)]
        self.velocities = [np.random.rand(self.population_size, dim) * 0.1 for _ in range(self.sub_populations)]
        self.best_particle_positions = [np.copy(p) for p in self.particles]
        self.best_particle_scores = [np.full(self.population_size, np.inf) for _ in range(self.sub_populations)]
        self.global_best_position = None
        self.global_best_score = np.inf
        self.fitness_evaluations = 0
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_min = 0.2
        self.w_max = 1.2
        self.turbulence_factor = 0.05
        self.f_min = 0.3
        self.f_max = 0.9
        self.cr = 0.9

    def initialize_population(self, dim):
        return np.random.rand(self.population_size, dim)

    def __call__(self, func):
        phase_change = self.budget // 3
        current_phase = 1
        while self.fitness_evaluations < self.budget:
            if self.fitness_evaluations // phase_change > current_phase:
                current_phase += 1
                self.adjust_parameters(current_phase)
            
            for sub_idx in range(self.sub_populations):
                self.update_particles(func, sub_idx)
                self.apply_self_adaptive_mutation(func, sub_idx)
                if np.random.rand() < 0.3:
                    self.apply_dynamic_levy_perturbation(func, sub_idx)
        return self.global_best_position

    def adjust_parameters(self, phase):
        self.c1 = max(self.c1 - 0.2, 1.5)
        self.c2 = min(self.c2 + 0.2, 2.5)
        self.w_max = max(self.w_max - 0.1, 0.8)
        self.turbulence_factor = min(self.turbulence_factor + 0.01, 0.1)

    def update_particles(self, func, sub_idx):
        w = self.w_max - (self.w_max - self.w_min) * (self.fitness_evaluations / self.budget)
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            score = func(self.particles[sub_idx][i])
            self.fitness_evaluations += 1

            if score < self.best_particle_scores[sub_idx][i]:
                self.best_particle_scores[sub_idx][i] = score
                self.best_particle_positions[sub_idx][i] = self.particles[sub_idx][i].copy()

            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[sub_idx][i].copy()

        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            velocity_cognitive = self.c1 * r1 * (self.best_particle_positions[sub_idx][i] - self.particles[sub_idx][i])
            velocity_social = self.c2 * r2 * (self.global_best_position - self.particles[sub_idx][i])
            self.velocities[sub_idx][i] = w * self.velocities[sub_idx][i] + velocity_cognitive + velocity_social
            self.particles[sub_idx][i] += self.velocities[sub_idx][i] + self.turbulence_factor * np.random.randn(self.dim)
            self.particles[sub_idx][i] = np.clip(self.particles[sub_idx][i], func.bounds.lb, func.bounds.ub)

    def apply_self_adaptive_mutation(self, func, sub_idx):
        diversity = np.mean(np.std(self.particles[sub_idx], axis=0))
        f = self.f_min + (self.f_max - self.f_min) * (diversity / (np.max([diversity, 1e-6])))

        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c, d = np.random.choice(indices, 4, replace=False)
            mutant = self.particles[sub_idx][a] + f * (self.particles[sub_idx][b] - self.particles[sub_idx][c])
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, self.particles[sub_idx][i])

            score = func(trial)
            self.fitness_evaluations += 1

            if score < self.best_particle_scores[sub_idx][i]:
                self.particles[sub_idx][i] = trial
                self.best_particle_scores[sub_idx][i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = trial

    def apply_dynamic_levy_perturbation(self, func, sub_idx):
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            lam = 1.5 - 0.4 * (self.fitness_evaluations / self.budget)
            levy_step = self.levy_flight(lam)
            candidate = self.particles[sub_idx][i] + levy_step
            candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)

            score = func(candidate)
            self.fitness_evaluations += 1

            if score < self.best_particle_scores[sub_idx][i]:
                self.particles[sub_idx][i] = candidate
                self.best_particle_scores[sub_idx][i] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = candidate

    def levy_flight(self, lam=1.3):
        sigma_u = np.power((np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2)) /
                           (np.math.gamma((1 + lam) / 2) * lam * np.power(2, ((lam - 1) / 2))), 1 / lam)
        u = np.random.normal(0, sigma_u, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.power(np.abs(v), 1 / lam)
        return step