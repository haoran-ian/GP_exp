import numpy as np

class RefinedAdaptiveChaoticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w_max = 0.9  # Inertia weight initial
        self.w_min = 0.4
        self.F_max = 0.9  # Differential mutation factor initial
        self.F_min = 0.2
        self.CR = 0.9  # Crossover probability
        self.chaos_iterations = 1000
        self.chaos_sequence = self.generate_chaotic_sequence(self.chaos_iterations)

    def generate_chaotic_sequence(self, length):
        x = 0.7  # initial value
        r = 3.9  # logistic map parameter
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
            # Adaptive parameter control
            progress = evaluations / self.budget
            self.w = self.w_max - (self.w_max - self.w_min) * progress
            self.F = self.F_max - (self.F_max - self.F_min) * progress
            
            # Apply chaos to cognitive and social components
            chaos_c1 = self.c1 * self.chaos_sequence[chaos_idx % self.chaos_iterations]
            chaos_c2 = self.c2 * self.chaos_sequence[(chaos_idx + self.chaos_iterations // 2) % self.chaos_iterations]
            chaos_idx += 1

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities +
                          chaos_c1 * r1 * (personal_best_positions - positions) +
                          chaos_c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            if evaluations % 100 == 0:  # Dynamic local search intensification
                intensification_scale = self.w * (ub - lb) * np.random.rand()
                positions = np.random.uniform(lb, ub, (self.population_size, self.dim)) + intensification_scale

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