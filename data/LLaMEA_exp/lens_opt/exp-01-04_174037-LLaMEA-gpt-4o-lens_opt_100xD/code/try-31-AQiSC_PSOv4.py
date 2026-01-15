import numpy as np

class AQiSC_PSOv4:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w = 0.9
        self.w_min = 0.4
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.alpha = 0.1
        self.beta = 0.05  # New parameter for adaptive influence
        self.particles_pos = np.random.rand(self.num_particles, self.dim)
        self.particles_vel = np.random.rand(self.num_particles, self.dim) * 0.1
        self.p_best_pos = np.copy(self.particles_pos)
        self.p_best_val = np.full(self.num_particles, np.inf)
        self.g_best_pos = np.zeros(self.dim)
        self.g_best_val = np.inf
        self.memory_matrix = np.full((self.num_particles, self.dim), np.inf)

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

                if error < self.memory_matrix[i].mean():
                    self.memory_matrix[i] = self.particles_pos[i]

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                c1 = self.c1_initial * (1 - eval_count / self.budget)
                c2 = self.c2_initial * (eval_count / self.budget)
                
                cognitive = c1 * r1 * (self.p_best_pos[i] - self.particles_pos[i])
                social = c2 * r2 * (self.g_best_pos - self.particles_pos[i])
                memory_influence = 0.1 * (np.average(self.memory_matrix, axis=0) - self.particles_pos[i])

                # Adaptive influence based on fitness
                adaptive_influence = self.beta * (1 - (self.p_best_val[i] / np.max(self.p_best_val))) * (ub - lb)

                self.particles_vel[i] = self.w * self.particles_vel[i] + cognitive + social + memory_influence + adaptive_influence

                velocity_correction = 0.1 * (self.p_best_pos[i] - self.particles_pos[i])
                self.particles_vel[i] += velocity_correction

                quantum_jump = 0.05 * (ub - lb) * np.random.uniform(-1, 1, self.dim)
                self.particles_pos[i] += self.particles_vel[i] + quantum_jump
                self.particles_pos[i] -= self.alpha * np.sign(self.particles_pos[i] - self.g_best_pos) * np.abs(self.particles_pos[i] - self.g_best_pos)

                self.particles_pos[i] = np.clip(self.particles_pos[i], lb, ub)
                self.particles_vel[i] = np.clip(self.particles_vel[i], -0.2, 0.2)

            self.w = self.w_min + (0.9 - self.w_min) * ((self.budget - eval_count) / self.budget)
            elite_influence = 0.05 * (self.g_best_pos - np.average(self.particles_pos, axis=0))
            self.particles_pos += elite_influence

            if eval_count % 10 == 0:
                self.w = 0.9 - (0.5 * (eval_count / self.budget))

        return self.g_best_pos