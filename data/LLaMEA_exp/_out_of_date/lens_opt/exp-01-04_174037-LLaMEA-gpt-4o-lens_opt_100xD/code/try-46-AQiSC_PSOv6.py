import numpy as np

class AQiSC_PSOv6:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.num_swarms = 3
        self.w = 0.9
        self.w_min = 0.4
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.alpha = 0.1
        self.particles_pos = np.random.rand(self.num_particles, self.dim)
        self.particles_vel = np.random.rand(self.num_particles, self.dim) * 0.1
        self.p_best_pos = np.copy(self.particles_pos)
        self.p_best_val = np.full(self.num_particles, np.inf)
        self.g_best_pos = np.zeros(self.dim)
        self.g_best_val = np.inf
        self.memory_matrix = np.full((self.num_particles, self.dim), np.inf)
        self.vel_bound = 0.2
        
        # Initialize swarm indices
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

                if error < self.memory_matrix[i].mean():
                    self.memory_matrix[i] = self.particles_pos[i]

            for swarm in self.swarm_indices:
                # Calculate local best within each swarm
                swarm_best_idx = np.argmin(self.p_best_val[swarm])
                swarm_best_pos = self.p_best_pos[swarm][swarm_best_idx]

                for i in swarm:
                    r1, r2 = np.random.rand(), np.random.rand()
                    c1 = self.c1_initial * (1 - eval_count / self.budget)
                    c2 = self.c2_initial * (eval_count / self.budget)
                    
                    cognitive = c1 * r1 * (self.p_best_pos[i] - self.particles_pos[i])
                    social = c2 * r2 * (self.g_best_pos - self.particles_pos[i])
                    swarm_influence = 0.1 * (swarm_best_pos - self.particles_pos[i])
                    memory_influence = 0.1 * (np.average(self.memory_matrix, axis=0) - self.particles_pos[i])

                    self.particles_vel[i] = self.w * self.particles_vel[i] + cognitive + social + memory_influence + swarm_influence
                    
                    velocity_correction = 0.1 * (self.p_best_pos[i] - self.particles_pos[i])
                    self.particles_vel[i] += velocity_correction
                    
                    adaptive_quantum_jump = 0.05 * (ub - lb) * np.random.uniform(-1, 1, self.dim) * (self.budget - eval_count) / self.budget
                    self.particles_pos[i] += self.particles_vel[i] + adaptive_quantum_jump
                    self.particles_pos[i] -= self.alpha * np.sign(self.particles_pos[i] - self.g_best_pos) * np.abs(self.particles_pos[i] - self.g_best_pos)

                    self.particles_pos[i] = np.clip(self.particles_pos[i], lb, ub)
                    
                    # Adaptive velocity clipping
                    adaptive_vel_bound = self.vel_bound * (1 - eval_count / self.budget)
                    self.particles_vel[i] = np.clip(self.particles_vel[i], -adaptive_vel_bound, adaptive_vel_bound)

            self.w = self.w_min + (0.9 - self.w_min) * ((self.budget - eval_count) / self.budget)
            elite_influence = 0.1 * (self.g_best_pos - np.average(self.particles_pos, axis=0))
            self.particles_pos += elite_influence
            
            # Clustering adjustment every 20 evaluations
            if eval_count % 20 == 0:
                cluster_center = np.average(self.particles_pos, axis=0)
                for i in range(self.num_particles):
                    self.particles_pos[i] += 0.05 * (cluster_center - self.particles_pos[i])

            # Particle diversity preservation strategy
            diversity_threshold = 0.1 * (ub - lb)
            particle_distances = np.linalg.norm(self.particles_pos - np.mean(self.particles_pos, axis=0), axis=1)
            if np.std(particle_distances) < diversity_threshold:
                self.particles_pos += 0.1 * (np.random.rand(self.num_particles, self.dim) * (ub - lb) - (ub - lb) / 2)

            # Elite particle reinforcement every 50 evaluations
            if eval_count % 50 == 0:
                elite_particle_strength = 0.2
                for i in range(self.num_particles):
                    if np.linalg.norm(self.particles_pos[i] - self.g_best_pos) < 0.5 * np.linalg.norm(ub - lb):
                        self.particles_pos[i] += elite_particle_strength * (self.g_best_pos - self.particles_pos[i])

        return self.g_best_pos