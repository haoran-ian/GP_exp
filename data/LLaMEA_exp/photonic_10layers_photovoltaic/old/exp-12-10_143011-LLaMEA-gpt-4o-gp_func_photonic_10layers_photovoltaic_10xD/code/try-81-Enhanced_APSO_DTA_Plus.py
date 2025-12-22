import numpy as np

class Enhanced_APSO_DTA_Plus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.initial_w = 0.9  # Initial inertia weight
        self.final_w = 0.4  # Final inertia weight
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.local_search_prob = 0.1  # Probability of performing local search
        self.particles = np.random.rand(self.pop_size, dim)
        self.velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.pop_size, np.inf)
        self.best_global_position = np.zeros(dim)
        self.best_global_score = np.inf
        self.func_eval_count = 0

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        self.particles = bounds_lb + self.particles * (bounds_ub - bounds_lb)
        
        while self.func_eval_count < self.budget:
            w = self.initial_w - (self.func_eval_count / self.budget) * (self.initial_w - self.final_w)

            for i in range(self.pop_size):
                current_score = func(self.particles[i])
                self.func_eval_count += 1
                
                if current_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = current_score
                    self.best_personal_positions[i] = self.particles[i].copy()
                
                if current_score < self.best_global_score:
                    self.best_global_score = current_score
                    self.best_global_position = self.particles[i].copy()
            
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            neighborhood_size = max(1, self.pop_size // 5)
            
            for i in range(self.pop_size):
                neighbors = np.random.choice(self.pop_size, neighborhood_size, replace=False)
                best_neighbor = neighbors[np.argmin(self.best_personal_scores[neighbors])]
                
                self.velocities[i] = (w * self.velocities[i] +
                                      self.c1 * r1[i] * (self.best_personal_positions[i] - self.particles[i]) +
                                      self.c2 * r2[i] * (self.best_personal_positions[best_neighbor] - self.particles[i]))
            
            self.particles += self.velocities
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)

            if np.random.rand() < self.local_search_prob:
                best_particle_idx = np.argmin(self.best_personal_scores)
                local_search_radius = 0.1 * (bounds_ub - bounds_lb)
                local_search_point = self.particles[best_particle_idx] + np.random.uniform(-local_search_radius, local_search_radius, self.dim)
                local_search_point = np.clip(local_search_point, bounds_lb, bounds_ub)
                local_search_score = func(local_search_point)
                self.func_eval_count += 1
                if local_search_score < self.best_global_score:
                    self.best_global_score = local_search_score
                    self.best_global_position = local_search_point

        return self.best_global_position, self.best_global_score