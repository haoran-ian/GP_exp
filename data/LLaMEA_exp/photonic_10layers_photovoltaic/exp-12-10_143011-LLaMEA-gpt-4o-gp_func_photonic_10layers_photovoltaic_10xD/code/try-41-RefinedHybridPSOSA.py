import numpy as np

class RefinedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 30
        self.pop_size = self.initial_pop_size
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.temp_factor = 0.99
        self.particles = np.random.rand(self.pop_size, dim)
        self.velocities = np.random.rand(self.pop_size, dim) * 0.1
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
            personal_dist = np.linalg.norm(self.best_personal_positions - self.particles, axis=1, keepdims=True)
            global_dist = np.linalg.norm(self.best_global_position - self.particles, axis=1, keepdims=True)
            adaptive_w = 0.4 + 0.5 * (1 - self.func_eval_count / self.budget)
            self.velocities = (adaptive_w * self.velocities +
                               self.c1 * r1 * (self.best_personal_positions - self.particles) / (personal_dist + 1e-6) +
                               self.c2 * r2 * (self.best_global_position - self.particles) / (global_dist + 1e-6))
            self.particles += self.velocities
            
            temp = self.temp_factor ** (self.func_eval_count / self.budget)
            perturbation = np.random.normal(0, temp, (self.pop_size, self.dim))
            self.particles += perturbation
            
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)
            
            if self.func_eval_count % (self.budget // 10) == 0:
                self.pop_size = max(10, self.pop_size // 2)
                self.particles = self.particles[:self.pop_size]
                self.velocities = self.velocities[:self.pop_size]
                self.best_personal_positions = self.best_personal_positions[:self.pop_size]
                self.best_personal_scores = self.best_personal_scores[:self.pop_size]
        
        return self.best_global_position, self.best_global_score