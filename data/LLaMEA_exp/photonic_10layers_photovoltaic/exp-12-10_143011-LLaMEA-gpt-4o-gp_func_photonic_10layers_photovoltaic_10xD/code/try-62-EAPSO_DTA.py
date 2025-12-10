import numpy as np

class EAPSO_DTA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.initial_w = 0.9  # Initial inertia weight
        self.final_w = 0.4    # Final inertia weight
        self.c1 = 2.0         # Cognitive component
        self.c2 = 2.0         # Social component
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
        
        # Normalize particles to the function's bounds
        self.particles = bounds_lb + self.particles * (bounds_ub - bounds_lb)
        
        while self.func_eval_count < self.budget:
            for i in range(self.pop_size):
                current_score = func(self.particles[i])
                self.func_eval_count += 1
                
                # Update personal best
                if current_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = current_score
                    self.best_personal_positions[i] = self.particles[i].copy()
                
                # Update global best
                if current_score < self.best_global_score:
                    self.best_global_score = current_score
                    self.best_global_position = self.particles[i].copy()
            
            # Update velocities and positions with adaptive coefficients
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            adapt_c1 = self.c1 * (1 - (self.func_eval_count / self.budget)**2)
            adapt_c2 = self.c2 * ((self.func_eval_count / self.budget)**2)
            inertia_weight = self.initial_w - (self.initial_w - self.final_w) * (self.func_eval_count / self.budget)
            personal_dist = np.linalg.norm(self.best_personal_positions - self.particles, axis=1, keepdims=True)
            global_dist = np.linalg.norm(self.best_global_position - self.particles, axis=1, keepdims=True)
            self.velocities = (inertia_weight * self.velocities +
                               adapt_c1 * r1 * (self.best_personal_positions - self.particles) / (personal_dist + 1e-6) +
                               adapt_c2 * r2 * (self.best_global_position - self.particles) / (global_dist + 1e-6))
            self.particles += self.velocities
            
            # Dynamic temperature annealing inspired perturbation
            temp = self.initial_temp * ((self.final_temp / self.initial_temp) ** (self.func_eval_count / self.budget))
            perturbation = np.random.normal(0, temp, (self.pop_size, self.dim))
            self.particles += perturbation
            
            # Keep particles inside bounds
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)
        
        return self.best_global_position, self.best_global_score