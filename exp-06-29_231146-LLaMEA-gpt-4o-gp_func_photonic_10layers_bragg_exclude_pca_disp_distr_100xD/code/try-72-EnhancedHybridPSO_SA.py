import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30  # Number of particles in the swarm
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1 = 1.5  # Cognitive (particle) weight
        self.c2 = 1.5  # Social (swarm) weight
        self.temperature = 1.0  # Initial temperature for SA
        self.cooling_rate = 0.98  # Cooling rate for SA
        self.eval_count = 0

    def adaptive_inertia_weight(self):
        # Adaptive inertia weight
        return self.w_max - ((self.w_max - self.w_min) * (self.eval_count / self.budget))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pos = np.random.uniform(low=lb, high=ub, size=(self.num_particles, self.dim))
        vel = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(self.num_particles, self.dim))
        pbest_pos = np.copy(pos)
        pbest_val = np.array([func(p) for p in pos])
        gbest_pos = pbest_pos[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)
        
        self.eval_count = self.num_particles

        while self.eval_count < self.budget:
            w = self.adaptive_inertia_weight()
            
            # Update velocities and positions
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            vel = w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            pos = pos + vel
            pos = np.clip(pos, lb, ub)
            
            # Evaluate new positions
            f_values = np.array([func(p) for p in pos])
            self.eval_count += self.num_particles
            
            # Update personal and global bests
            better_mask = f_values < pbest_val
            pbest_pos = np.where(better_mask[:, np.newaxis], pos, pbest_pos)
            pbest_val = np.where(better_mask, f_values, pbest_val)
            
            if np.min(f_values) < gbest_val:
                gbest_val = np.min(f_values)
                gbest_pos = pos[np.argmin(f_values)]
            
            # Simulated Annealing acceptance criterion
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

            # Optional local search phase to exploit around the global best
            if self.eval_count < self.budget:
                local_search_pos = gbest_pos + np.random.uniform(-0.05, 0.05, self.dim)
                local_search_pos = np.clip(local_search_pos, lb, ub)
                local_search_val = func(local_search_pos)
                self.eval_count += 1
                if local_search_val < gbest_val:
                    gbest_val = local_search_val
                    gbest_pos = local_search_pos

            # Cooling schedule
            self.temperature *= self.cooling_rate

        return gbest_pos, gbest_val