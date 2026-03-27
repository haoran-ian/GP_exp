import numpy as np

class Enhanced_AQiSC_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.c3_initial = 1.0  # New component for neighborhood influence
        self.alpha = 0.1
        self.particles_pos = np.random.rand(self.num_particles, self.dim)
        self.particles_vel = np.random.rand(self.num_particles, self.dim) * 0.1
        self.p_best_pos = np.copy(self.particles_pos)
        self.p_best_val = np.full(self.num_particles, np.inf)
        self.g_best_pos = np.zeros(self.dim)
        self.g_best_val = np.inf
        self.neighbors = np.random.randint(0, self.num_particles, size=(self.num_particles, 2))  # Define neighbors

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
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()  # New random component
                c1 = self.c1_initial * (1 - eval_count / self.budget)
                c2 = self.c2_initial * (eval_count / self.budget)

                # Dynamic Neighborhood Influence
                neighborhood_best = min(self.p_best_val[self.neighbors[i]])
                c3 = self.c3_initial * (1 - eval_count / self.budget)

                cognitive = c1 * r1 * (self.p_best_pos[i] - self.particles_pos[i])
                social = c2 * r2 * (self.g_best_pos - self.particles_pos[i])
                neighborhood = c3 * r3 * (neighborhood_best - self.particles_pos[i])

                self.particles_vel[i] = self.w_min + (self.w_max - self.w_min) * ((self.budget - eval_count) / self.budget)
                self.particles_vel[i] *= self.particles_vel[i] + cognitive + social + neighborhood

                self.particles_pos[i] += self.particles_vel[i]
                self.particles_pos[i] = np.clip(self.particles_pos[i], lb, ub)

        return self.g_best_pos