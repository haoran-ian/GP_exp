import numpy as np

class QISC_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.w = 0.9
        self.w_min = 0.4
        self.alpha = 0.1
        self.q_amplitude = 0.5  # quantum amplitude
        self.particles_pos = np.random.rand(self.num_particles, self.dim)
        self.particles_vel = np.random.rand(self.num_particles, self.dim) * 0.1
        self.p_best_pos = np.copy(self.particles_pos)
        self.p_best_val = np.full(self.num_particles, np.inf)
        self.g_best_pos = np.zeros(self.dim)
        self.g_best_val = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        eval_count = 0

        self.particles_pos = lb + (ub - lb) * self.particles_pos

        while eval_count < self.budget:
            for i in range(self.num_particles):
                error = func(self.particles_pos[i])
                eval_count += 1

                if error < self.p_best_val[i]:
                    self.p_best_val[i] = error
                    self.p_best_pos[i] = self.particles_pos[i]

                if error < self.g_best_val:
                    self.g_best_val = error
                    self.g_best_pos = self.particles_pos[i]

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()

                c1 = self.c1_initial * (1 - eval_count / self.budget)
                c2 = self.c2_initial * (eval_count / self.budget)

                cognitive = c1 * r1 * (self.p_best_pos[i] - self.particles_pos[i])
                social = c2 * r2 * (self.g_best_pos - self.particles_pos[i])
                self.particles_vel[i] = self.w * self.particles_vel[i] + cognitive + social

                velocity_correction = 0.1 * (self.p_best_pos[i] - self.particles_pos[i])
                self.particles_vel[i] += velocity_correction

                quantum_exploration = self.q_amplitude * np.sin(np.random.rand(self.dim) * np.pi)
                self.particles_pos[i] += self.particles_vel[i] + quantum_exploration

                self.particles_pos[i] = np.clip(self.particles_pos[i], lb, ub)

            self.w = self.w_min + (0.9 - self.w_min) * ((self.budget - eval_count) / self.budget)
            self.q_amplitude *= 0.99

        return self.g_best_pos