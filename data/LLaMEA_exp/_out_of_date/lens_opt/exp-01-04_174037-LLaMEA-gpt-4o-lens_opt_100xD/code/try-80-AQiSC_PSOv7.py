import numpy as np

class AQiSC_PSOv7:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 40  # Increased number of particles
        self.num_swarms = 4  # Increased number of swarms
        self.w_initial = 0.85  # Slightly adjusted initial inertia
        self.w_final = 0.3  # Lower final inertia for faster convergence
        self.c1_initial = 2.0  # Adjusted learning coefficients for balance
        self.c2_initial = 2.0
        self.alpha = 0.05  # Reduced alpha for less aggressive global attraction
        self.particles_pos = np.random.rand(self.num_particles, self.dim)
        self.particles_vel = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        self.p_best_pos = np.copy(self.particles_pos)
        self.p_best_val = np.full(self.num_particles, np.inf)
        self.g_best_pos = np.zeros(self.dim)
        self.g_best_val = np.inf
        self.memory_matrix = np.full((self.num_particles, self.dim), np.inf)
        self.vel_bound = 0.1  # Decreased velocity bound for stability
        self.swarm_indices = np.array_split(np.arange(self.num_particles), self.num_swarms)

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
                # Update memory with decayed influence
                memory_update_factor = 0.8  # Decay factor for memory update
                self.memory_matrix[i] = (memory_update_factor * self.memory_matrix[i] + 
                                         (1 - memory_update_factor) * self.particles_pos[i])

            for swarm in self.swarm_indices:
                swarm_best_idx = np.argmin(self.p_best_val[swarm])
                swarm_best_pos = self.p_best_pos[swarm][swarm_best_idx]

                for i in swarm:
                    r1, r2 = np.random.rand(), np.random.rand()
                    c1 = self.c1_initial * (1 - eval_count / self.budget)
                    c2 = self.c2_initial * (eval_count / self.budget)
                    cognitive = c1 * r1 * (self.p_best_pos[i] - self.particles_pos[i])
                    social = c2 * r2 * (self.g_best_pos - self.particles_pos[i])
                    neighborhood_influence = 0.2 * (swarm_best_pos - self.particles_pos[i])
                    memory_influence = 0.1 * (self.memory_matrix[i] - self.particles_pos[i])

                    self.particles_vel[i] = (self.inertia_weight(eval_count) * self.particles_vel[i] + 
                                             cognitive + social + memory_influence + neighborhood_influence)

                    self.particles_pos[i] += self.particles_vel[i]
                    self.particles_pos[i] = np.clip(self.particles_pos[i], lb, ub)

                    adaptive_vel_bound = self.vel_bound * (1 - eval_count / self.budget)
                    self.particles_vel[i] = np.clip(self.particles_vel[i], -adaptive_vel_bound, adaptive_vel_bound)

            elite_influence = 0.05 * (self.g_best_pos - np.average(self.particles_pos, axis=0))
            self.particles_pos += elite_influence

            if eval_count % 20 == 0:
                cluster_center = np.average(self.particles_pos, axis=0)
                for i in range(self.num_particles):
                    self.particles_pos[i] += 0.05 * (cluster_center - self.particles_pos[i])

        return self.g_best_pos

    def inertia_weight(self, eval_count):
        return self.w_final + (self.w_initial - self.w_final) * ((self.budget - eval_count) / self.budget)