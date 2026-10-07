import numpy as np

class RefinedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight
        self.c1 = 2.0  # Cognitive weight
        self.c2 = 2.0  # Social weight
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Faster cooling rate for SA

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pos = np.random.uniform(low=lb, high=ub, size=(self.num_particles, self.dim))
        vel = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(self.num_particles, self.dim))
        pbest_pos = np.copy(pos)
        pbest_val = np.array([func(p) for p in pos])
        gbest_pos = pbest_pos[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)
        
        eval_count = self.num_particles

        while eval_count < self.budget:
            # Dynamic inertia weight adjustment
            inertia_weight = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            vel = inertia_weight * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            pos = pos + vel
            pos = np.clip(pos, lb, ub)
            
            f_values = np.array([func(p) for p in pos])
            eval_count += self.num_particles

            better_mask = f_values < pbest_val
            pbest_pos = np.where(better_mask[:, np.newaxis], pos, pbest_pos)
            pbest_val = np.where(better_mask, f_values, pbest_val)

            if np.min(f_values) < gbest_val:
                gbest_val = np.min(f_values)
                gbest_pos = pos[np.argmin(f_values)]

            for i in range(self.num_particles):
                new_pos = pos[i] + np.random.uniform(-0.05, 0.05, self.dim) * self.temperature
                new_pos = np.clip(new_pos, lb, ub)
                new_val = func(new_pos)
                eval_count += 1
                if new_val < f_values[i] or np.random.rand() < np.exp((f_values[i] - new_val) / self.temperature):
                    pos[i] = new_pos
                    f_values[i] = new_val
                    if new_val < gbest_val:
                        gbest_val = new_val
                        gbest_pos = new_pos

            self.temperature *= self.cooling_rate

        return gbest_pos, gbest_val