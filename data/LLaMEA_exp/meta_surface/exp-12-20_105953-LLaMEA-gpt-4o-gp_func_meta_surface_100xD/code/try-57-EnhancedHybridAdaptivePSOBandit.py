import numpy as np

class EnhancedHybridAdaptivePSOBandit:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(self.dim))
        self.c1 = 1.5
        self.c2 = 2.5
        self.w_min = 0.3
        self.w_max = 0.8
        self.neighborhood_size = max(2, int(self.pop_size * 0.3))
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.arm_rewards = np.zeros(2)  # Two strategies: DE mutation and Levy flight
        self.arm_counts = np.ones(2)  # Initial count to avoid division by zero
        self.explore_probability = 0.1

    def levy_flight(self, dim, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def select_arm(self):
        total_counts = np.sum(self.arm_counts)
        ucb_values = self.arm_rewards / self.arm_counts + np.sqrt(2 * np.log(total_counts) / self.arm_counts)
        return np.argmax(ucb_values)

    def update_arm(self, arm, reward):
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        value = self.arm_rewards[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.arm_rewards[arm] = new_value

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
            
            if np.random.rand() < self.explore_probability:
                arm = self.select_arm()
            else:
                arm = np.random.choice(2)
            
            if arm == 0:  # DE mutation
                f = 0.5 + np.random.rand() * 0.5
                indices = np.random.choice(self.pop_size, 5, replace=False)
                mutant_positions = self.positions[indices[0]] + f * (self.positions[indices[1]] - self.positions[indices[2]] +
                                                                     self.positions[indices[3]] - self.positions[indices[4]])
                mutant_positions = np.clip(mutant_positions, lb, ub)
                mutant_score = func(mutant_positions)
                eval_count += 1
                improvement = self.personal_best_scores[indices[0]] - mutant_score
                if mutation_score < self.personal_best_scores[indices[0]]:
                    self.personal_best_scores[indices[0]] = mutant_score
                    self.personal_best_positions[indices[0]] = mutant_positions
                if mutant_score < self.global_best_score:
                    self.global_best_score = mutant_score
                    self.global_best_position = mutant_positions

            elif arm == 1:  # Levy flight
                step = self.levy_flight(self.dim)
                levy_positions = self.global_best_position + step
                levy_positions = np.clip(levy_positions, lb, ub)
                levy_score = func(levy_positions)
                eval_count += 1
                improvement = self.global_best_score - levy_score
                if levy_score < self.global_best_score:
                    self.global_best_score = levy_score
                    self.global_best_position = levy_positions
            
            self.update_arm(arm, improvement)

            r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (local_best_positions - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.positions += self.velocities
            
            self.positions = np.clip(self.positions, lb, ub)
        
        return self.global_best_position, self.global_best_score