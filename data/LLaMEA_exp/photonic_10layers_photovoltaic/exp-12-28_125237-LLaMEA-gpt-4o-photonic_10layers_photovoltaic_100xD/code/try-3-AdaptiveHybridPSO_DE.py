import numpy as np

class AdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.best_particle_positions = np.copy(self.particles)
        self.best_particle_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.fitness_evaluations = 0
    
    def levy_flight(self, lam=1.5):
        # Levy flight step
        sigma_u = np.power((np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2)) / 
                           (np.math.gamma((1 + lam) / 2) * lam * np.power(2, ((lam - 1) / 2))), 1 / lam)
        u = np.random.normal(0, sigma_u, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.power(np.abs(v), 1 / lam)
        return step
    
    def __call__(self, func):
        while self.fitness_evaluations < self.budget:
            self.update_particles(func)
            self.apply_differential_evolution(func)
            if np.random.rand() < 0.3:
                self.apply_levy_perturbation(func)
        return self.global_best_position

    def update_particles(self, func):
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
            self.velocities[i] = (0.5 * self.velocities[i] +
                                  1.5 * r1 * (self.best_particle_positions[i] - self.particles[i]) +
                                  1.5 * r2 * (self.global_best_position - self.particles[i]))
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], func.bounds.lb, func.bounds.ub)

    def apply_differential_evolution(self, func):
        for i in range(self.population_size):
            if self.fitness_evaluations >= self.budget:
                break

            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.particles[a] + 0.8 * (self.particles[b] - self.particles[c])
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

            cross_points = np.random.rand(self.dim) < 0.9
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