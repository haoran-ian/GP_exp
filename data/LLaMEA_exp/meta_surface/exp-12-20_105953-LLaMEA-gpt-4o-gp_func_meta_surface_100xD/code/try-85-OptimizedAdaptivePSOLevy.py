import numpy as np

class OptimizedAdaptivePSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(self.dim))
        self.c1 = 1.5
        self.c2 = 2.5
        self.w_min = 0.3
        self.w_max = 0.8
        self.positions = np.random.uniform(0, 1, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
    
    def levy_flight(self, dim, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        eval_count = 0
        
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                
                score = func(self.positions[i])
                eval_count += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = np.copy(self.positions[i])
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.positions[i])
            
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            
            r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.positions += self.velocities
            
            self.positions = np.clip(self.positions, lb, ub)
            
            if np.random.rand() < 0.1:
                elite_step = self.levy_flight(self.dim)
                elite_position = self.global_best_position + elite_step
                elite_position = np.clip(elite_position, lb, ub)
                elite_score = func(elite_position)
                eval_count += 1
                if elite_score < self.global_best_score:
                    self.global_best_score = elite_score
                    self.global_best_position = elite_position
        
        return self.global_best_position, self.global_best_score