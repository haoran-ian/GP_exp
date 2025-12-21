import numpy as np

class ImprovedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(self.dim))  # Population size
        self.c1 = 1.5  # Lower Cognitive coefficient for better global exploration
        self.c2 = 2.5  # Higher Social coefficient for improved convergence
        self.w_min = 0.4  # Min inertia weight
        self.w_max = 0.9  # Max inertia weight
        self.neighborhood_size = max(4, self.pop_size // 5)  # Adjusted dynamic neighborhood size
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
                
                # Evaluate current position
                score = func(self.positions[i])
                eval_count += 1
                
                # Update personal bests
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                    
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
                    
            # Dynamic neighborhood topology
            local_best_positions = np.copy(self.personal_best_positions)
            for i in range(self.pop_size):
                neighbors = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                best_neighbor = min(neighbors, key=lambda x: self.personal_best_scores[x])
                local_best_positions[i] = self.personal_best_positions[best_neighbor]
            
            # Adaptive adjustment of inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            
            # Opposition-based learning strategy
            if np.random.rand() < 0.1:  # 10% chance to explore opposition solutions
                opposite_positions = lb + ub - self.positions
                opposite_scores = np.array([func(pos) for pos in opposite_positions])
                eval_count += len(opposite_positions)
                improved_indices = opposite_scores < self.personal_best_scores
                self.personal_best_scores[improved_indices] = opposite_scores[improved_indices]
                self.personal_best_positions[improved_indices] = opposite_positions[improved_indices]
                if np.min(opposite_scores) < self.global_best_score:
                    self.global_best_score = np.min(opposite_scores)
                    self.global_best_position = opposite_positions[np.argmin(opposite_scores)]

            # Update velocities and positions
            r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (local_best_positions - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.positions += self.velocities
            
            # Clip to bounds
            self.positions = np.clip(self.positions, lb, ub)
        
        return self.global_best_position, self.global_best_score