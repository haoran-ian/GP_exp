import numpy as np

class RefinedAdaptivePSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(self.dim))
        self.c1 = 1.5
        self.c2 = 2.5
        self.w_min = 0.2
        self.w_max = 0.9
        self.neighborhood_size = max(2, int(self.pop_size * 0.3))
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.num_subpops = 3
        self.subpop_size = self.pop_size // self.num_subpops
        
    def levy_flight(self, dim, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def adaptive_mutation(self, candidate, lb, ub):
        step_size = np.random.uniform(0.1, 0.5) * (ub - lb)
        mutation = candidate + step_size * np.random.normal(size=self.dim)
        return np.clip(mutation, lb, ub)

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
                dynamic_neighborhood = max(2, int(self.pop_size * 0.3 * (1 - eval_count / self.budget)))
                neighbors = np.random.choice(self.pop_size, dynamic_neighborhood, replace=False)
                best_neighbor = min(neighbors, key=lambda x: self.personal_best_scores[x])
                local_best_positions[i] = self.personal_best_positions[best_neighbor]
            
            w = self.w_max - (self.w_max - self.w_min) * (np.sqrt(eval_count / self.budget))
            
            if np.random.rand() < 0.15:
                f = 0.5 + np.random.rand() * 0.5
                indices = np.random.choice(self.pop_size, 5, replace=False)
                mutant_positions = self.positions[indices[0]] + f * (self.positions[indices[1]] - self.positions[indices[2]] +
                                                                     self.positions[indices[3]] - self.positions[indices[4]])
                mutant_positions = np.clip(mutant_positions, lb, ub)
                mutant_score = func(mutant_positions)
                eval_count += 1
                if mutant_score < self.personal_best_scores[indices[0]]:
                    self.personal_best_scores[indices[0]] = mutant_score
                    self.personal_best_positions[indices[0]] = mutant_positions
                if mutant_score < self.global_best_score:
                    self.global_best_score = mutant_score
                    self.global_best_position = mutant_positions
            
            r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (local_best_positions - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.positions += self.velocities
            
            self.positions = np.clip(self.positions, lb, ub)
            
            if np.random.rand() < 0.1:
                step = self.levy_flight(self.dim)
                levy_positions = self.global_best_position + step
                levy_positions = np.clip(levy_positions, lb, ub)
                levy_score = func(levy_positions)
                eval_count += 1
                if levy_score < self.global_best_score:
                    self.global_best_score = levy_score
                    self.global_best_position = levy_positions
            
            if eval_count % (self.budget // 10) == 0:
                for subpop in range(self.num_subpops):
                    start = subpop * self.subpop_size
                    end = start + self.subpop_size
                    best_subpop_idx = np.argmin(self.personal_best_scores[start:end])
                    best_subpop_position = self.personal_best_positions[start + best_subpop_idx]
                    for i in range(self.pop_size):
                        if np.random.rand() < 0.05:
                            self.positions[i] = best_subpop_position
                        if np.random.rand() < 0.05:
                            self.positions[i] = self.adaptive_mutation(self.positions[i], lb, ub)
        
        return self.global_best_position, self.global_best_score