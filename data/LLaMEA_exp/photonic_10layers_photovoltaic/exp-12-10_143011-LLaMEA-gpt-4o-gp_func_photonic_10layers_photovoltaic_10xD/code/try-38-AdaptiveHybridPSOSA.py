import numpy as np

class AdaptiveHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 30
        self.max_pop_size = 50
        self.min_pop_size = 10
        self.population_growth_factor = 0.1
        self.w_min = 0.2
        self.w_max = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.temp_factor = 0.99
        self.particles = np.random.rand(self.initial_pop_size, dim)
        self.velocities = np.random.rand(self.initial_pop_size, dim) * 0.1
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.initial_pop_size, np.inf)
        self.best_global_position = np.zeros(dim)
        self.best_global_score = np.inf
        self.func_eval_count = 0

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        
        # Normalize particles to the function's bounds
        self.particles = bounds_lb + self.particles * (bounds_ub - bounds_lb)
        
        while self.func_eval_count < self.budget:
            pop_size = len(self.particles)
            w = self.w_max - ((self.w_max - self.w_min) * (self.func_eval_count / self.budget))
            
            for i in range(pop_size):
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
            r1 = np.random.rand(pop_size, self.dim)
            r2 = np.random.rand(pop_size, self.dim)
            personal_dist = np.linalg.norm(self.best_personal_positions - self.particles, axis=1, keepdims=True)
            global_dist = np.linalg.norm(self.best_global_position - self.particles, axis=1, keepdims=True)
            self.velocities = (w * self.velocities +
                               self.c1 * r1 * (self.best_personal_positions - self.particles) / (personal_dist + 1e-6) +
                               self.c2 * r2 * (self.best_global_position - self.particles) / (global_dist + 1e-6))
            self.particles += self.velocities
            
            # Simulated Annealing inspired perturbation
            temp = self.temp_factor ** (self.func_eval_count / self.budget)
            perturbation = np.random.normal(0, temp, (pop_size, self.dim))
            self.particles += perturbation
            
            # Keep particles inside bounds
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)

            # Dynamic population adjustment
            new_pop_size = int(self.initial_pop_size + self.population_growth_factor * (self.func_eval_count / self.budget) * (self.max_pop_size - self.initial_pop_size))
            new_pop_size = max(self.min_pop_size, new_pop_size)
            new_pop_size = min(self.max_pop_size, new_pop_size)
            if new_pop_size != pop_size:
                self.particles = np.resize(self.particles, (new_pop_size, self.dim))
                self.velocities = np.resize(self.velocities, (new_pop_size, self.dim))
                self.best_personal_positions = np.resize(self.best_personal_positions, (new_pop_size, self.dim))
                self.best_personal_scores = np.resize(self.best_personal_scores, new_pop_size)
                if new_pop_size > pop_size:
                    self.particles[pop_size:] = np.random.rand(new_pop_size - pop_size, self.dim)
                    self.velocities[pop_size:] = np.random.rand(new_pop_size - pop_size, self.dim) * 0.1
                    self.best_personal_scores[pop_size:] = np.inf
        
        return self.best_global_position, self.best_global_score