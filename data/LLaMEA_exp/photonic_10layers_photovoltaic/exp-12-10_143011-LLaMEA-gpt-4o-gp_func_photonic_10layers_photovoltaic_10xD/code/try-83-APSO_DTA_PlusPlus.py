import numpy as np

class APSO_DTA_PlusPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.w = 0.5  # Inertia weight
        self.c1_start = 2.0  # Start cognitive component
        self.c2_start = 2.0  # Start social component
        self.c1_end = 0.5  # End cognitive component
        self.c2_end = 0.5  # End social component
        self.initial_temp = 1.0  # Initial temperature for annealing
        self.final_temp = 0.01  # Final temperature for annealing
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
            scores = np.zeros(self.pop_size)
            
            for i in range(self.pop_size):
                current_score = func(self.particles[i])
                self.func_eval_count += 1
                scores[i] = current_score
                
                # Update personal best
                if current_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = current_score
                    self.best_personal_positions[i] = self.particles[i].copy()
                
                # Update global best
                if current_score < self.best_global_score:
                    self.best_global_score = current_score
                    self.best_global_position = self.particles[i].copy()
            
            # Sort particles by their scores
            sorted_indices = np.argsort(scores)
            self.particles = self.particles[sorted_indices]
            self.best_personal_positions = self.best_personal_positions[sorted_indices]
            self.velocities = self.velocities[sorted_indices]
            self.best_personal_scores = self.best_personal_scores[sorted_indices]

            # Update learning rates
            adapt_c1 = self.c1_start + (self.c1_end - self.c1_start) * (self.func_eval_count / self.budget)
            adapt_c2 = self.c2_start + (self.c2_end - self.c2_start) * (self.func_eval_count / self.budget)

            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            # Stochastic ranking and adaptive velocity update
            for i in range(self.pop_size):
                self.velocities[i] = (self.w * self.velocities[i] +
                                      adapt_c1 * r1[i] * (self.best_personal_positions[i] - self.particles[i]) +
                                      adapt_c2 * r2[i] * (self.best_global_position - self.particles[i]))

            self.particles += self.velocities
            
            # Dynamic temperature annealing inspired perturbation
            temp = self.initial_temp * ((self.final_temp / self.initial_temp) ** (self.func_eval_count / self.budget))
            perturbation = np.random.normal(0, temp, (self.pop_size, self.dim))
            self.particles += perturbation
            
            # Keep particles inside bounds
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)
        
        return self.best_global_position, self.best_global_score