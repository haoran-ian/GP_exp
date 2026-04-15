import numpy as np

class HybridPSOwithADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.7   # Inertia weight
        self.F = 0.5   # Differential mutation factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize positions and velocities
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
            # Ensure positions are within bounds
            positions = np.clip(positions, lb, ub)
            
            # Evaluate the current population
            scores = np.array([func(x) for x in positions])
            evaluations += self.population_size

            # Update personal and global bests
            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = positions[improved]
            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

            # Apply adaptive differential mutation to enhance diversity
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
            self.w = 0.5 + 0.2 * np.random.rand()  # Dynamic inertia weight
            self.CR = 0.8 + 0.2 * np.random.rand()  # Dynamic crossover probability

        return global_best_position, personal_best_scores[global_best_index]