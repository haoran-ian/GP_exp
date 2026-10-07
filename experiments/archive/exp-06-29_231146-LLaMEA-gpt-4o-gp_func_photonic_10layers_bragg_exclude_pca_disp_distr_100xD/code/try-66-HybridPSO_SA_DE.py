import numpy as np

class HybridPSO_SA_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30  # Number of particles in the swarm
        self.w_max, self.w_min = 0.9, 0.4  # Dynamic inertia weights
        self.c1_init, self.c2_init = 2.5, 0.5  # Initial cognitive and social weights
        self.temperature = 1.0  # Initial temperature for SA
        self.cooling_rate = 0.98  # Cooling rate for SA
        self.F = 0.5  # Differential evolution scaling factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize particles
        lb, ub = func.bounds.lb, func.bounds.ub
        pos = np.random.uniform(low=lb, high=ub, size=(self.num_particles, self.dim))
        vel = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(self.num_particles, self.dim))
        pbest_pos = np.copy(pos)
        pbest_val = np.array([func(p) for p in pos])
        gbest_pos = pbest_pos[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)
        
        eval_count = self.num_particles
        
        while eval_count < self.budget:
            # Dynamic parameter adjustment
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            c1 = self.c1_init - (self.c1_init - 1.5) * (eval_count / self.budget)
            c2 = self.c2_init + (2.0 - self.c2_init) * (eval_count / self.budget)
            
            # Update velocities and positions
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
            pos = pos + vel
            pos = np.clip(pos, lb, ub)  # Ensure positions are within bounds
            
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
            
            # Differential Evolution mutation and crossover
            for i in range(self.num_particles):
                idxs = [idx for idx in range(self.num_particles) if idx != i]
                a, b, c = pos[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pos[i])
                trial_val = func(trial)
                eval_count += 1
                if trial_val < f_values[i]:
                    f_values[i] = trial_val
                    pos[i] = trial
                    if trial_val < gbest_val:
                        gbest_val = trial_val
                        gbest_pos = trial
            
            # Cooling schedule
            self.temperature *= self.cooling_rate

        return gbest_pos, gbest_val