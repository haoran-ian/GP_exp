import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w_max = 0.9  # Maximum inertia weight for adaptive strategy
        self.w_min = 0.4  # Minimum inertia weight for adaptive strategy
        self.c1 = 1.5
        self.c2 = 1.5
        self.temperature = 1.0
        self.cooling_rate_initial = 0.995  # Initial cooling rate
        self.cooling_rate_final = 0.9  # Final cooling rate
        self.cooling_schedule_length = 0.5  # Fraction of budget for cooling schedule
        
    def adapt_inertia(self, iter, max_iter):
        """ Adaptive inertia weight strategy """
        return self.w_max - ((self.w_max - self.w_min) * (iter / max_iter))
        
    def dynamic_cooling_rate(self, eval_count, max_eval):
        """ Dynamic cooling rate strategy """
        phase_ratio = self.cooling_schedule_length * max_eval
        if eval_count < phase_ratio:
            return self.cooling_rate_initial
        else:
            return self.cooling_rate_final
            
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pos = np.random.uniform(low=lb, high=ub, size=(self.num_particles, self.dim))
        vel = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(self.num_particles, self.dim))
        pbest_pos = np.copy(pos)
        pbest_val = np.array([func(p) for p in pos])
        gbest_pos = pbest_pos[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)
        
        eval_count = self.num_particles
        iter_count = 0
        
        while eval_count < self.budget:
            # Update velocities and positions
            w = self.adapt_inertia(iter_count, self.budget // self.num_particles)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            vel = w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            pos = pos + vel
            pos = np.clip(pos, lb, ub)
            
            # Evaluate new positions
            f_values = np.array([func(p) for p in pos])
            eval_count += self.num_particles
            
            # Update personal and global bests
            better_mask = f_values < pbest_val
            pbest_pos = np.where(better_mask[:, np.newaxis], pos, pbest_pos)
            pbest_val = np.where(better_mask, f_values, pbest_val)
            
            if np.min(f_values) < gbest_val:
                gbest_val = np.min(f_values)
                gbest_pos = pos[np.argmin(f_values)]
            
            # Simulated Annealing acceptance criterion
            cooling_rate = self.dynamic_cooling_rate(eval_count, self.budget)
            for i in range(self.num_particles):
                new_pos = pos[i] + np.random.uniform(-0.1, 0.1, self.dim) * self.temperature
                new_pos = np.clip(new_pos, lb, ub)
                new_val = func(new_pos)
                eval_count += 1
                if new_val < f_values[i] or np.random.rand() < np.exp((f_values[i] - new_val) / self.temperature):
                    pos[i] = new_pos
                    f_values[i] = new_val
                    if new_val < gbest_val:
                        gbest_val = new_val
                        gbest_pos = new_pos
            
            # Cooling schedule
            self.temperature *= cooling_rate
            iter_count += 1

        return gbest_pos, gbest_val