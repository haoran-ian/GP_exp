import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30  # Number of particles in the swarm
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive (particle) weight
        self.c2 = 1.5  # Social (swarm) weight
        self.temperature = 1.0  # Initial temperature for SA
        self.cooling_rate = 0.99  # Cooling rate for SA
        self.mutation_factor = 0.8  # Differential mutation factor
        self.mutation_prob = 0.1  # Mutation probability

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
            
            for i in range(self.num_particles):
                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(self.num_particles, 3, replace=False)
                    x1, x2, x3 = pos[indices[0]], pos[indices[1]], pos[indices[2]]
                    mutant = x1 + self.mutation_factor * (x2 - x3)
                    mutant = np.clip(mutant, lb, ub)
                    mutant_val = func(mutant)
                    eval_count += 1
                    if mutant_val < f_values[i]:
                        pos[i] = mutant
                        f_values[i] = mutant_val
                        if mutant_val < gbest_val:
                            gbest_val = mutant_val
                            gbest_pos = mutant
            
            self.temperature *= self.cooling_rate

        return gbest_pos, gbest_val