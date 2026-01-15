import numpy as np

class GE_SCPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1_initial = 2.0  # initial cognitive coefficient
        self.c2_initial = 2.0  # initial social coefficient
        self.w_initial = 0.9   # initial inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.alpha_initial = 0.1  # initial gradient descent step size
        self.alpha_min = 0.01  # minimum step size
        self.particles_pos = np.random.rand(self.num_particles, self.dim)
        self.particles_vel = np.random.rand(self.num_particles, self.dim) * 0.1
        self.p_best_pos = np.copy(self.particles_pos)
        self.p_best_val = np.full(self.num_particles, np.inf)
        self.g_best_pos = np.zeros(self.dim)
        self.g_best_val = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        eval_count = 0

        # Initialize positions within the bounds
        self.particles_pos = lb + (ub - lb) * self.particles_pos

        while eval_count < self.budget:
            for i in range(self.num_particles):
                error = func(self.particles_pos[i])
                eval_count += 1

                # Update personal best
                if error < self.p_best_val[i]:
                    self.p_best_val[i] = error
                    self.p_best_pos[i] = self.particles_pos[i]

                # Update global best
                if error < self.g_best_val:
                    self.g_best_val = error
                    self.g_best_pos = self.particles_pos[i]

            # Update velocities and positions
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()

                # Dynamic cognitive and social coefficients
                c1 = self.c1_initial * (1 - eval_count / self.budget)
                c2 = self.c2_initial * (eval_count / self.budget)

                cognitive = c1 * r1 * (self.p_best_pos[i] - self.particles_pos[i])
                social = c2 * r2 * (self.g_best_pos - self.particles_pos[i])
                self.particles_vel[i] = self.w_initial * self.particles_vel[i] + cognitive + social

                # Quadratic self-correcting mechanism for velocity
                velocity_correction = 0.2 * (self.p_best_pos[i] - self.particles_pos[i]) ** 2
                self.particles_vel[i] += velocity_correction

                # Adaptive gradient descent update
                adaptive_alpha = self.alpha_initial - (self.alpha_initial - self.alpha_min) * (eval_count / self.budget)
                gradient = np.gradient(self.particles_pos[i])
                self.particles_pos[i] += self.particles_vel[i] - adaptive_alpha * gradient

                # Ensure the particles remain within bounds
                self.particles_pos[i] = np.clip(self.particles_pos[i], lb, ub)

            # Adaptive inertia weight update
            self.w_initial = self.w_min + (0.9 - self.w_min) * ((self.budget - eval_count) / self.budget)

        return self.g_best_pos