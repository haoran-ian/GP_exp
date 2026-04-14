import numpy as np

class EnhancedHybridPSOwithSelfAdaptation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_start, self.c1_end = 1.5, 2.5
        self.c2_start, self.c2_end = 1.5, 2.5
        self.w_start, self.w_end = 0.9, 0.4
        self.F_start, self.F_end = 0.9, 0.2
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        evaluations = self.population_size

        while evaluations < self.budget:
            c1 = self.c1_start + (self.c1_end - self.c1_start) * (evaluations / self.budget)
            c2 = self.c2_start + (self.c2_end - self.c2_start) * (evaluations / self.budget)
            w = self.w_start - (self.w_start - self.w_end) * (evaluations / self.budget)
            F = self.F_start - (self.F_start - self.F_end) * (evaluations / self.budget)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = w * velocities + c1 * r1 * (personal_best_positions - positions) + c2 * r2 * (global_best_position - positions)
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
                    mutant = x0 + F * (x1 - x2)
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

        return global_best_position, personal_best_scores[global_best_index]