import numpy as np

class APSO_DTA_Plus_Plus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.initial_temp = 1.0
        self.final_temp = 0.01
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
            adapt_c1 = self.c1 * (1 - np.exp(-self.func_eval_count / self.budget))
            adapt_c2 = self.c2 * np.exp(-self.func_eval_count / self.budget)
            personal_dist = np.linalg.norm(self.best_personal_positions - self.particles, axis=1, keepdims=True)
            global_dist = np.linalg.norm(self.best_global_position - self.particles, axis=1, keepdims=True)
            
            neighborhood_size = max(1, self.pop_size // 5)
            for i in range(self.pop_size):
                neighbors = np.random.choice(self.pop_size, neighborhood_size, replace=False)
                best_neighbor = neighbors[np.argmin(self.best_personal_scores[neighbors])]
                # Adjust velocity with adaptive learning and hybrid perturbation
                self.velocities[i] = (self.w * self.velocities[i] +
                                      adapt_c1 * r1[i] * (self.best_personal_positions[i] - self.particles[i]) / (personal_dist[i] + 1e-6) +
                                      adapt_c2 * r2[i] * (self.best_personal_positions[best_neighbor] - self.particles[i]) / (global_dist[i] + 1e-6))
            
            self.particles += self.velocities
            
            temp = self.initial_temp * ((self.final_temp / self.initial_temp) ** (self.func_eval_count / self.budget))
            perturbation = np.random.normal(0, temp, (self.pop_size, self.dim))
            hybrid_perturbation = perturbation + 0.5 * np.random.rand(self.pop_size, self.dim) * np.sign(self.best_global_position - self.particles)
            self.particles += hybrid_perturbation
            
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)
        
        return self.best_global_position, self.best_global_score