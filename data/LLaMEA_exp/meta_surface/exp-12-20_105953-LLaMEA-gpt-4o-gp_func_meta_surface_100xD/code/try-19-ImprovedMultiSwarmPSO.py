import numpy as np

class ImprovedMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(self.dim))  # Population size
        self.num_swarms = 2  # Number of swarms
        self.positions = [np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim)) for _ in range(self.num_swarms)]
        self.velocities = [np.random.uniform(-1, 1, (self.pop_size, self.dim)) for _ in range(self.num_swarms)]
        self.personal_best_positions = [np.copy(p) for p in self.positions]
        self.personal_best_scores = [np.full(self.pop_size, float('inf')) for _ in range(self.num_swarms)]
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.c1_max, self.c1_min = 2.5, 1.5
        self.c2_max, self.c2_min = 2.5, 1.5
        self.w_min, self.w_max = 0.4, 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        eval_count = 0
        
        while eval_count < self.budget:
            for swarm_idx in range(self.num_swarms):
                for i in range(self.pop_size):
                    if eval_count >= self.budget:
                        break
                    
                    # Evaluate current position
                    score = func(self.positions[swarm_idx][i])
                    eval_count += 1
                    
                    # Update personal bests
                    if score < self.personal_best_scores[swarm_idx][i]:
                        self.personal_best_scores[swarm_idx][i] = score
                        self.personal_best_positions[swarm_idx][i] = self.positions[swarm_idx][i]
                    
                    # Update global best
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[swarm_idx][i]
                
                # Adaptive adjustment of coefficients
                progress = eval_count / self.budget
                c1 = self.c1_max - (self.c1_max - self.c1_min) * progress
                c2 = self.c2_min + (self.c2_max - self.c2_min) * progress
                w = self.w_max - (self.w_max - self.w_min) * progress

                # Random orthogonal exploration technique
                if np.random.rand() < 0.2:  # 20% chance for orthogonal exploration
                    ortho_positions = self.positions[swarm_idx] + np.random.rand(*self.positions[swarm_idx].shape) * (ub - lb) / 5
                    ortho_scores = np.array([func(pos) for pos in ortho_positions])
                    eval_count += len(ortho_positions)
                    improved_indices = ortho_scores < self.personal_best_scores[swarm_idx]
                    self.personal_best_scores[swarm_idx][improved_indices] = ortho_scores[improved_indices]
                    self.personal_best_positions[swarm_idx][improved_indices] = ortho_positions[improved_indices]
                    if np.min(ortho_scores) < self.global_best_score:
                        self.global_best_score = np.min(ortho_scores)
                        self.global_best_position = ortho_positions[np.argmin(ortho_scores)]
                
                # Update velocities and positions
                r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
                r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
                cognitive_velocity = c1 * r1 * (self.personal_best_positions[swarm_idx] - self.positions[swarm_idx])
                social_velocity = c2 * r2 * (self.global_best_position - self.positions[swarm_idx])
                self.velocities[swarm_idx] = w * self.velocities[swarm_idx] + cognitive_velocity + social_velocity
                self.positions[swarm_idx] += self.velocities[swarm_idx]
                
                # Clip to bounds
                self.positions[swarm_idx] = np.clip(self.positions[swarm_idx], lb, ub)
        
        return self.global_best_position, self.global_best_score