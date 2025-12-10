import numpy as np

class EnhancedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight
        self.c1_initial = 2.0  # Initial cognitive component
        self.c2_initial = 2.0  # Initial social component
        self.temp_factor = 0.99  # Cooling factor for Simulated Annealing
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
        
        # Normalize particles to the function's bounds
        self.particles = bounds_lb + self.particles * (bounds_ub - bounds_lb)
        
        while self.func_eval_count < self.budget:
            inertia_weight = self.w_max - (
                (self.w_max - self.w_min) * (self.func_eval_count / self.budget)
            )
            c1 = self.c1_initial - (
                (self.c1_initial - 0.5) * (self.func_eval_count / self.budget)
            )
            c2 = self.c2_initial - (
                (self.c2_initial - 0.5) * (self.func_eval_count / self.budget)
            )
            
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
            
            # Update velocities and positions
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.velocities = (inertia_weight * self.velocities +
                               c1 * r1 * (self.best_personal_positions - self.particles) +
                               c2 * r2 * (self.best_global_position - self.particles))
            self.particles += self.velocities
            
            # Simulated Annealing inspired perturbation
            temp = self.temp_factor ** (self.func_eval_count / self.budget)
            perturbation = np.random.normal(0, temp, (self.pop_size, self.dim))
            self.particles += perturbation
            
            # Keep particles inside bounds
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)
        
        return self.best_global_position, self.best_global_score