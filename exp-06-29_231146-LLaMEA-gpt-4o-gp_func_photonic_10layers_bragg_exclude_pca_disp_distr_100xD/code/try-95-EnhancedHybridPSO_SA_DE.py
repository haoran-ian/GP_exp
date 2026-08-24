import numpy as np

class EnhancedHybridPSO_SA_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.de_factor = 0.8  # Differential Evolution scaling factor
        self.cr = 0.7  # Crossover probability for DE

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
            # Particle Swarm Optimization part
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

            # Differential Evolution part
            for i in range(self.num_particles):
                indices = np.random.choice(self.num_particles, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.num_particles, 3, replace=False)
                a, b, c = pos[indices]
                mutant = np.clip(a + self.de_factor * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, pos[i])
                trial_val = func(trial)
                eval_count += 1
                if trial_val < f_values[i]:
                    pos[i] = trial
                    f_values[i] = trial_val
                    if trial_val < gbest_val:
                        gbest_val = trial_val
                        gbest_pos = trial
            
            # Simulated Annealing with dynamic cooling
            for i in range(self.num_particles):
                new_pos = pos[i] + np.random.uniform(-0.1, 0.1, self.dim) * self.temperature
                new_pos = np.clip(new_pos, lb, ub)
                new_val = func(new_pos)
                eval_count += 1
                acceptance_criterion = (new_val < f_values[i]) or np.random.rand() < np.exp((f_values[i] - new_val) / self.temperature)
                if acceptance_criterion:
                    pos[i] = new_pos
                    f_values[i] = new_val
                    if new_val < gbest_val:
                        gbest_val = new_val
                        gbest_pos = new_pos
            
            # Dynamic cooling rate
            self.temperature *= (self.cooling_rate + (0.01 * (eval_count / self.budget)))

        return gbest_pos, gbest_val