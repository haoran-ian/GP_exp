import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(self.dim))  
        self.c1_min, self.c1_max = 1.0, 2.0
        self.c2_min, self.c2_max = 2.0, 3.0
        self.w_min, self.w_max = 0.4, 0.9
        self.neighborhood_size = max(3, self.pop_size // 5)
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
                    
            local_best_positions = np.copy(self.personal_best_positions)
            for i in range(self.pop_size):
                neighbors = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                best_neighbor = min(neighbors, key=lambda x: self.personal_best_scores[x])
                local_best_positions[i] = self.personal_best_positions[best_neighbor]
            
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            
            if np.random.rand() < 0.1:
                opposite_positions = lb + ub - self.positions
                opposite_scores = np.array([func(pos) for pos in opposite_positions])
                eval_count += len(opposite_positions)
                improved_indices = opposite_scores < self.personal_best_scores
                self.personal_best_scores[improved_indices] = opposite_scores[improved_indices]
                self.personal_best_positions[improved_indices] = opposite_positions[improved_indices]
                if np.min(opposite_scores) < self.global_best_score:
                    self.global_best_score = np.min(opposite_scores)
                    self.global_best_position = opposite_positions[np.argmin(opposite_scores)]

            diversity = np.std(self.positions, axis=0).mean()
            c1 = self.c1_min + (self.c1_max - self.c1_min) * (1 - diversity)
            c2 = self.c2_min + (self.c2_max - self.c2_min) * diversity
            
            r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = c2 * r2 * (local_best_positions - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.positions += self.velocities
            
            self.positions = np.clip(self.positions, lb, ub)
        
        return self.global_best_position, self.global_best_score