import numpy as np

class AdaptiveMemoryHybridPSOwithADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w_min = 0.4
        self.w_max = 0.9
        self.F_min = 0.2
        self.F_max = 0.9
        self.CR = 0.9  # Crossover probability
        self.memory_size = 10  # Memory for historical data
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        
        memory_positions = np.copy(personal_best_positions[:self.memory_size])
        memory_scores = np.copy(personal_best_scores[:self.memory_size])
        
        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive inertia weight and mutation factor
            self.w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
            self.F = self.F_max - ((self.F_max - self.F_min) * (evaluations / self.budget))

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, lb, ub)
            
            scores = np.array([func(x) for x in positions])
            evaluations += self.population_size

            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = positions[improved]
            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

            for i in range(self.population_size):
                if np.random.rand() < self.CR:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = positions[indices[0]], positions[indices[1]], positions[indices[2]]
                    mutant = x0 + self.F * (x1 - x2)
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, positions[i])
                    trial = np.clip(trial, lb, ub)
                    trial_score = func(trial)
                    evaluations += 1
                    if trial_score < scores[i]:
                        positions[i] = trial
                        scores[i] = trial_score
                        if trial_score < personal_best_scores[i]:
                            personal_best_scores[i] = trial_score
                            personal_best_positions[i] = trial
                            if trial_score < personal_best_scores[global_best_index]:
                                global_best_position = trial

            # Update memory with best positions and scores
            combined_positions = np.vstack((memory_positions, personal_best_positions))
            combined_scores = np.hstack((memory_scores, personal_best_scores))
            best_indices = np.argsort(combined_scores)[:self.memory_size]
            memory_positions = combined_positions[best_indices]
            memory_scores = combined_scores[best_indices]

        return global_best_position, personal_best_scores[global_best_index]