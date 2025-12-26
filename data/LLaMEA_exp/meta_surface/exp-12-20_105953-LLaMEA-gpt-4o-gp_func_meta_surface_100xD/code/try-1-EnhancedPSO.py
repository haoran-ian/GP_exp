import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(self.dim))  # Population size
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, float('inf'))
        
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                    
                score = func(self.positions[i])
                eval_count += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                    
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
                    
            # Adaptive inertia weight
            self.w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            
            # Update velocities and positions
            r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive_velocity + social_velocity
            
            # Velocity clamping
            vmax = 0.1 * (ub - lb)
            self.velocities = np.clip(self.velocities, -vmax, vmax)
            
            self.positions += self.velocities
            
            # Clip to bounds
            self.positions = np.clip(self.positions, lb, ub)
        
        return self.global_best_position, self.global_best_score