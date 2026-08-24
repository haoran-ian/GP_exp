import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_num_particles = 30  # Initial number of particles in the swarm
        self.w_min, self.w_max = 0.3, 0.9  # Adaptive inertia weight bounds
        self.c1 = 1.5  # Cognitive (particle) weight
        self.c2 = 1.5  # Social (swarm) weight
        self.temperature = 1.0  # Initial temperature for SA
        self.cooling_rate = 0.99  # Cooling rate for SA
        self.eval_count = 0

    def _evaluate_particles(self, pos, func):
        f_values = np.array([func(p) for p in pos])
        self.eval_count += len(pos)
        return f_values

    def __call__(self, func):
        # Initialize particles
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = self.initial_num_particles
        pos = np.random.uniform(low=lb, high=ub, size=(num_particles, self.dim))
        vel = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(num_particles, self.dim))
        pbest_pos = np.copy(pos)
        pbest_val = self._evaluate_particles(pos, func)
        gbest_pos = pbest_pos[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)
        
        while self.eval_count < self.budget:
            # Dynamic population resizing
            if self.eval_count / self.budget > 0.5 and num_particles > 10:
                num_particles = max(10, num_particles // 2)
                pos, vel = pos[:num_particles], vel[:num_particles]
                pbest_pos, pbest_val = pbest_pos[:num_particles], pbest_val[:num_particles]

            # Update velocities and positions
            w = self.w_max - (self.w_max - self.w_min) * (self.eval_count / self.budget)
            r1, r2 = np.random.rand(num_particles, self.dim), np.random.rand(num_particles, self.dim)
            vel = w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            pos = pos + vel
            pos = np.clip(pos, lb, ub)  # Ensure positions are within bounds
            
            # Evaluate new positions
            f_values = self._evaluate_particles(pos, func)
            
            # Update personal and global bests
            better_mask = f_values < pbest_val
            pbest_pos = np.where(better_mask[:, np.newaxis], pos, pbest_pos)
            pbest_val = np.where(better_mask, f_values, pbest_val)
            
            if np.min(f_values) < gbest_val:
                gbest_val = np.min(f_values)
                gbest_pos = pos[np.argmin(f_values)]
            
            # Simulated Annealing acceptance criterion
            for i in range(num_particles):
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
            
            # Cooling schedule
            self.temperature *= self.cooling_rate

        return gbest_pos, gbest_val