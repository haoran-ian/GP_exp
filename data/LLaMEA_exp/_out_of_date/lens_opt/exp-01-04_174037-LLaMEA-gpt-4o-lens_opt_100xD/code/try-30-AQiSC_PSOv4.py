import numpy as np

class AQiSC_PSOv4:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.alpha = 0.1
        self.particles_pos = np.random.rand(self.num_particles, self.dim)
        self.particles_vel = np.random.rand(self.num_particles, self.dim) * 0.1
        self.p_best_pos = np.copy(self.particles_pos)
        self.p_best_val = np.full(self.num_particles, np.inf)
        self.g_best_pos = np.zeros(self.dim)
        self.g_best_val = np.inf
        self.memory_matrix = np.full((self.num_particles, self.dim), np.inf)
        self.stagnation_counter = np.zeros(self.num_particles)
        self.stagnation_threshold = 20

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
                    self.stagnation_counter[i] = 0
                else:
                    self.stagnation_counter[i] += 1

                if error < self.g_best_val:
                    self.g_best_val = error
                    self.g_best_pos = self.particles_pos[i]

                if error < self.memory_matrix[i].mean():
                    self.memory_matrix[i] = self.particles_pos[i]

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = self.c1 * r1 * (self.p_best_pos[i] - self.particles_pos[i])
                social = self.c2 * r2 * (self.g_best_pos - self.particles_pos[i])
                memory_influence = 0.1 * (np.average(self.memory_matrix, axis=0) - self.particles_pos[i])

                self.particles_vel[i] = self.w * self.particles_vel[i] + cognitive + social + memory_influence

                velocity_correction = 0.1 * (self.p_best_pos[i] - self.particles_pos[i])
                self.particles_vel[i] += velocity_correction

                self.particles_pos[i] += self.particles_vel[i]
                self.particles_pos[i] -= self.alpha * np.sign(self.particles_pos[i] - self.g_best_pos) * np.abs(self.particles_pos[i] - self.g_best_pos)
                self.particles_pos[i] = np.clip(self.particles_pos[i], lb, ub)
                self.particles_vel[i] = np.clip(self.particles_vel[i], -0.2, 0.2)

                if self.stagnation_counter[i] >= self.stagnation_threshold:
                    self.particles_pos[i] = lb + (ub - lb) * np.random.rand(self.dim)
                    self.stagnation_counter[i] = 0

            self.w = self.w_min + (0.9 - self.w_min) * ((self.budget - eval_count) / self.budget)
            elite_influence = 0.05 * (self.g_best_pos - np.average(self.particles_pos, axis=0))
            self.particles_pos += elite_influence

            if eval_count % 10 == 0:
                self.w = 0.9 - (0.5 * (eval_count / self.budget))

        return self.g_best_pos