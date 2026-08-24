import numpy as np

class Enhanced_HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_init = 2.0
        self.c2_init = 2.0
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.eval_count = 0

    def update_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.eval_count / self.budget))

    def update_cognitive_social_weights(self):
        progress = self.eval_count / self.budget
        c1 = self.c1_init - progress * (self.c1_init - 1.5)
        c2 = self.c2_init + progress * (1.5 - self.c2_init)
        return c1, c2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pos = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        vel = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.num_particles, self.dim))
        pbest_pos = np.copy(pos)
        pbest_val = np.array([func(p) for p in pos])
        gbest_pos = pbest_pos[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)

        self.eval_count = self.num_particles

        while self.eval_count < self.budget:
            w = self.update_inertia_weight()
            c1, c2 = self.update_cognitive_social_weights()
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
            pos += vel
            pos = np.clip(pos, lb, ub)

            f_values = np.array([func(p) for p in pos])
            self.eval_count += self.num_particles

            better_mask = f_values < pbest_val
            pbest_pos[better_mask] = pos[better_mask]
            pbest_val[better_mask] = f_values[better_mask]

            if np.min(f_values) < gbest_val:
                gbest_val = np.min(f_values)
                gbest_pos = pos[np.argmin(f_values)]

            for i in range(self.num_particles):
                new_pos = pos[i] + np.random.uniform(-0.1, 0.1, self.dim) * self.temperature
                new_pos = np.clip(new_pos, lb, ub)
                new_val = func(new_pos)
                self.eval_count += 1
                if new_val < f_values[i] or np.random.rand() < np.exp((f_values[i] - new_val) / self.temperature):
                    pos[i] = new_pos
                    f_values[i] = new_val
                    if new_val < gbest_val:
                        gbest_val = new_val
                        gbest_pos = new_pos

            self.temperature *= self.cooling_rate

        return gbest_pos, gbest_val