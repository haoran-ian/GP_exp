import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w = 0.9  # Start with a higher inertia weight for wide exploration
        self.c1 = 2.0  # Increasing cognitive weight for personal learning
        self.c2 = 2.0  # Increasing social weight for group learning
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Faster cooling for quicker convergence
        self.elite_count = 5  # Number of elite particles for learning

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
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            vel = self.w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
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
            
            # Elite particle learning
            elite_indices = np.argsort(f_values)[:self.elite_count]
            elite_pos = pos[elite_indices]
            for i in range(self.num_particles):
                elite_choice = elite_pos[np.random.randint(self.elite_count)]
                proposed_pos = pos[i] + np.random.uniform(-0.1, 0.1, self.dim) * self.temperature * (elite_choice - pos[i])
                proposed_pos = np.clip(proposed_pos, lb, ub)
                proposed_val = func(proposed_pos)
                eval_count += 1
                if proposed_val < f_values[i] or np.random.rand() < np.exp((f_values[i] - proposed_val) / self.temperature):
                    pos[i] = proposed_pos
                    f_values[i] = proposed_val
                    if proposed_val < gbest_val:
                        gbest_val = proposed_val
                        gbest_pos = proposed_pos
            
            self.temperature *= self.cooling_rate
            self.w *= 0.99  # Gradually reduce inertia weight for fine-tuning

        return gbest_pos, gbest_val