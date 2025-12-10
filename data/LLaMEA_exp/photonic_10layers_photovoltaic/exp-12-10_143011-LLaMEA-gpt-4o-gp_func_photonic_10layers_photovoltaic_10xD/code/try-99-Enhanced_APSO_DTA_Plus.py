import numpy as np

class Enhanced_APSO_DTA_Plus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.initial_temp = 1.0  # Initial temperature for annealing
        self.final_temp = 0.01  # Final temperature for annealing
        self.particles = np.random.rand(self.pop_size, dim)
        self.velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.pop_size, np.inf)
        self.best_global_position = np.zeros(dim)
        self.best_global_score = np.inf
        self.func_eval_count = 0
        self.phase_switch = self.budget // 3
        self.dynamic_restart_threshold = 30  # Threshold for dynamic restart

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
            adapt_c1 = self.c1 * (1.0 - self.func_eval_count / self.budget)
            adapt_c2 = self.c2 * (self.func_eval_count / self.budget)

            if self.func_eval_count < self.phase_switch:
                velocity_weight = 0.9
            elif self.func_eval_count < 2 * self.phase_switch:
                velocity_weight = 0.5
            else:
                velocity_weight = 0.3

            for i in range(self.pop_size):
                inertia = velocity_weight * self.velocities[i]
                cognitive = adapt_c1 * r1[i] * (self.best_personal_positions[i] - self.particles[i])
                social = adapt_c2 * r2[i] * (self.best_global_position - self.particles[i])
                self.velocities[i] = inertia + cognitive + social

            self.particles += self.velocities

            temp = self.initial_temp * ((self.final_temp / self.initial_temp) ** (self.func_eval_count / self.budget))
            perturbation = np.random.normal(0, temp, (self.pop_size, self.dim))
            self.particles += perturbation

            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)

            if self.func_eval_count % self.dynamic_restart_threshold == 0:
                diverse_particles = np.random.rand(self.pop_size, self.dim)
                self.particles = np.where(np.random.rand(self.pop_size, self.dim) < 0.1, 
                                          bounds_lb + diverse_particles * (bounds_ub - bounds_lb), 
                                          self.particles)
        
        return self.best_global_position, self.best_global_score