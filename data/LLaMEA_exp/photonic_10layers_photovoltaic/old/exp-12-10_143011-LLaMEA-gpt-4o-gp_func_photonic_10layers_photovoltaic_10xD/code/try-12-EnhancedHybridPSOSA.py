import numpy as np

class EnhancedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_pop_size = 30
        self.max_pop_size = 50
        self.min_pop_size = 10
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.temp_factor = 0.99  # Cooling factor for Simulated Annealing
        self.particles = np.random.rand(self.init_pop_size, dim)
        self.velocities = np.random.rand(self.init_pop_size, dim) * 0.1
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.init_pop_size, np.inf)
        self.best_global_position = np.zeros(dim)
        self.best_global_score = np.inf
        self.func_eval_count = 0

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        
        # Normalize particles to the function's bounds
        self.particles = bounds_lb + self.particles * (bounds_ub - bounds_lb)
        
        while self.func_eval_count < self.budget:
            current_pop_size = int(self.min_pop_size + 
                                   (self.max_pop_size - self.min_pop_size) * 
                                   (1 - self.func_eval_count / self.budget))
            self.particles = self.particles[:current_pop_size]
            self.velocities = self.velocities[:current_pop_size]
            self.best_personal_positions = self.best_personal_positions[:current_pop_size]
            self.best_personal_scores = self.best_personal_scores[:current_pop_size]
            
            for i in range(current_pop_size):
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
            
            # Update velocities and positions with adaptive learning rates
            r1 = np.random.rand(current_pop_size, self.dim)
            r2 = np.random.rand(current_pop_size, self.dim)
            adaptive_c1 = self.c1 * (1 - self.func_eval_count / self.budget)
            adaptive_c2 = self.c2 * (self.func_eval_count / self.budget)
            self.velocities = (self.w * self.velocities +
                               adaptive_c1 * r1 * (self.best_personal_positions - self.particles) +
                               adaptive_c2 * r2 * (self.best_global_position - self.particles))
            self.particles += self.velocities
            
            # Simulated Annealing inspired perturbation
            temp = self.temp_factor ** (self.func_eval_count / self.budget)
            perturbation = np.random.normal(0, temp, (current_pop_size, self.dim))
            self.particles += perturbation
            
            # Keep particles inside bounds
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)
        
        return self.best_global_position, self.best_global_score