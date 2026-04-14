import numpy as np

class AdvancedHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_initial = 2.5  # Initial cognitive coefficient
        self.c2_initial = 0.5  # Initial social coefficient
        self.w = 0.9   # Inertia weight initial
        self.w_min = 0.4
        self.w_max = 0.9
        self.F1 = 0.8   # Differential mutation factor 1
        self.F2 = 0.6   # Differential mutation factor 2
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            self.w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
            self.c1 = self.c1_initial - evaluations / self.budget  # Dynamically adjust c1
            self.c2 = self.c2_initial + evaluations / self.budget  # Dynamically adjust c2

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
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
                    mutant1 = x0 + self.F1 * (x1 - x2)
                    mutant2 = x0 + self.F2 * (x1 - x2)
                    trial1 = np.where(np.random.rand(self.dim) < self.CR, mutant1, positions[i])
                    trial2 = np.where(np.random.rand(self.dim) < self.CR, mutant2, positions[i])
                    trial1 = np.clip(trial1, lb, ub)
                    trial2 = np.clip(trial2, lb, ub)
                    trial1_score = func(trial1)
                    trial2_score = func(trial2)
                    evaluations += 2
                    if trial1_score < scores[i]:
                        positions[i] = trial1
                        scores[i] = trial1_score
                        if trial1_score < personal_best_scores[i]:
                            personal_best_scores[i] = trial1_score
                            personal_best_positions[i] = trial1
                            if trial1_score < personal_best_scores[global_best_index]:
                                global_best_position = trial1
                    elif trial2_score < scores[i]:
                        positions[i] = trial2
                        scores[i] = trial2_score
                        if trial2_score < personal_best_scores[i]:
                            personal_best_scores[i] = trial2_score
                            personal_best_positions[i] = trial2
                            if trial2_score < personal_best_scores[global_best_index]:
                                global_best_position = trial2

        return global_best_position, personal_best_scores[global_best_index]