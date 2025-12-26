import numpy as np

class EnhancedAdaptiveDualPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(self.dim))  # Population size
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w_min = 0.4  # Min inertia weight
        self.w_max = 0.9  # Max inertia weight
        self.v_max = 0.2 * (func.bounds.ub - func.bounds.lb)  # Max velocity
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, float('inf'))
        
        eval_count = 0
        toggle_flag = True  # Toggle flag for dual population strategy
        while eval_count < self.budget:
            if toggle_flag:
                population = self.positions
            else:
                # Opposition-based population
                population = lb + ub - self.positions
            
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                    
                # Evaluate current position
                score = func(population[i])
                eval_count += 1
                
                # Update personal bests
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = population[i]
                    
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = population[i]
                    
            # Adaptive adjustment of inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            
            # Update velocities and positions
            r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            
            # Velocity clamping
            self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
            
            self.positions += self.velocities
            
            # Clip to bounds
            self.positions = np.clip(self.positions, lb, ub)
            
            # Toggle flag to switch between original and opposition population
            toggle_flag = not toggle_flag
        
        return self.global_best_position, self.global_best_score