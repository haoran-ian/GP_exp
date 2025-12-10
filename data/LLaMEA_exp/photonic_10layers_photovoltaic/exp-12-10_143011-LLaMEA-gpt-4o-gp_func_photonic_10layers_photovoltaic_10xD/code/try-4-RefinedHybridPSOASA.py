import numpy as np

class RefinedHybridPSOASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.init_temp = 1.0  # Initial temperature for ASA
        self.cooling_rate = 0.95  # Adaptive cooling rate
        self.particles = np.random.rand(self.pop_size, dim)
        self.velocities = np.random.rand(self.pop_size, dim) * 0.1
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.pop_size, np.inf)
        self.best_global_position = np.zeros(dim)
        self.best_global_score = np.inf
        self.func_eval_count = 0

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub

        self.particles = bounds_lb + self.particles * (bounds_ub - bounds_lb)

        while self.func_eval_count < self.budget:
            for i in range(self.pop_size):
                current_score = func(self.particles[i])
                self.func_eval_count += 1

                if current_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = current_score
                    self.best_personal_positions[i] = self.particles[i].copy()

                if current_score < self.best_global_score:
                    self.best_global_score = current_score
                    self.best_global_position = self.particles[i].copy()

            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.best_personal_positions - self.particles) +
                               self.c2 * r2 * (self.best_global_position - self.particles))
            self.particles += self.velocities

            temp = self.init_temp * (self.cooling_rate ** (self.func_eval_count / self.budget))
            perturbation = np.random.normal(0, temp, (self.pop_size, self.dim))
            self.particles += perturbation

            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)

            # Adaptive cooling rate adjustment based on progress
            if self.func_eval_count % (self.budget // 10) == 0:
                if self.best_global_score < np.min(self.best_personal_scores) * 1.01:  # Minimal improvement threshold
                    self.cooling_rate *= 0.9

        return self.best_global_position, self.best_global_score