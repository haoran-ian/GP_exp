import numpy as np

class AQiSC_PSOv4:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.w = 0.9
        self.w_min = 0.4
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.alpha = 0.1
        self.particles_pos = np.random.rand(self.num_particles, self.dim)
        self.particles_vel = np.random.rand(self.num_particles, self.dim) * 0.1
        self.p_best_pos = np.copy(self.particles_pos)
        self.p_best_val = np.full(self.num_particles, np.inf)
        self.g_best_pos = np.zeros(self.dim)
        self.g_best_val = np.inf
        
        # Entropy-based adaptive convergence zone
        self.entropy_threshold = 0.5
        self.convergence_zone_radius = 0.1

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

            # Compute entropy to adjust convergence zone
            particle_distances = np.linalg.norm(self.particles_pos - self.g_best_pos, axis=1)
            entropy = -np.sum((particle_distances / particle_distances.sum()) * np.log(particle_distances / particle_distances.sum() + 1e-9))
            
            if entropy < self.entropy_threshold:
                self.convergence_zone_radius *= 0.95  # Shrink convergence zone
            else:
                self.convergence_zone_radius = min(self.convergence_zone_radius * 1.05, (ub - lb) / 2)  # Expand convergence zone
            
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                c1 = self.c1_initial * (1 - eval_count / self.budget)
                c2 = self.c2_initial * (eval_count / self.budget)
                
                cognitive = c1 * r1 * (self.p_best_pos[i] - self.particles_pos[i])
                social = c2 * r2 * (self.g_best_pos - self.particles_pos[i])
                
                # Adaptive convergence zone influence
                zone_influence = self.convergence_zone_radius * (np.random.rand(self.dim) - 0.5)
                self.particles_vel[i] = self.w * self.particles_vel[i] + cognitive + social + zone_influence

                # Update position with boundaries
                self.particles_pos[i] += self.particles_vel[i]
                self.particles_pos[i] = np.clip(self.particles_pos[i], lb, ub)
                self.particles_vel[i] = np.clip(self.particles_vel[i], -0.2, 0.2)

            # Dynamic inertia weight adjustment
            self.w = self.w_min + (0.9 - self.w_min) * ((self.budget - eval_count) / self.budget)
            if eval_count % 10 == 0:
                self.w = 0.9 - (0.5 * (eval_count / self.budget))

        return self.g_best_pos