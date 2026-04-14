import numpy as np

class EnhancedAdaptiveChaoticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.F_max = 0.9
        self.F_min = 0.2
        self.CR = 0.9
        self.chaos_iterations = 1000
        self.chaos_sequence = self.generate_chaotic_sequence(self.chaos_iterations)
        self.dim_mutation_prob = 0.2

    def generate_chaotic_sequence(self, length):
        x = 0.7
        r = 3.9
        sequence = []
        for _ in range(length):
            x = r * x * (1 - x)
            sequence.append(x)
        return sequence

    def levy_flight(self, step_scale=0.01):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 1.5))
        return step_scale * step

    def gaussian_mutation(self, positions, lb, ub):
        mutation_indices = np.random.rand(self.population_size, self.dim) < self.dim_mutation_prob
        mutation_values = np.random.normal(0, 0.1, (self.population_size, self.dim))
        positions[mutation_indices] += mutation_values[mutation_indices]
        positions = np.clip(positions, lb, ub)
        return positions

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        evaluations = self.population_size
        chaos_idx = 0

        while evaluations < self.budget:
            self.w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
            self.F = self.F_max - ((self.F_max - self.F_min) * (evaluations / self.budget))

            chaos_c1 = self.c1 * self.chaos_sequence[(chaos_idx + evaluations) % self.chaos_iterations]
            chaos_c2 = self.c2 * self.chaos_sequence[(chaos_idx + self.chaos_iterations // 2) % self.chaos_iterations]
            chaos_idx += 1

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          chaos_c1 * r1 * (personal_best_positions - positions) +
                          chaos_c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            positions = self.gaussian_mutation(positions, lb, ub)

            if evaluations % 100 == 0:
                restart_prob = 0.05 + 0.05 * (evaluations / self.budget)
                if np.random.rand() < restart_prob:
                    positions = np.random.uniform(lb, ub, (self.population_size, self.dim))

            scores = np.array([func(x) for x in positions])
            evaluations += self.population_size

            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = positions[improved]
            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

            for i in range(self.population_size):
                if np.random.rand() < self.CR * (1 - evaluations / self.budget):
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = positions[indices[0]], positions[indices[1]], positions[indices[2]]
                    mutant = x0 + self.F * (x1 - x2) + self.levy_flight()
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