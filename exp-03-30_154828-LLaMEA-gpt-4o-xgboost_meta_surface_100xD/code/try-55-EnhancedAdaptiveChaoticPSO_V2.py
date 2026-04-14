import numpy as np

class EnhancedAdaptiveChaoticPSO_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.min_population_size = 20
        self.max_population_size = 100
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w_max = 0.9  # Inertia weight initial
        self.w_min = 0.4
        self.F_max = 0.9  # Differential mutation factor initial
        self.F_min = 0.2
        self.CR = 0.9  # Crossover probability
        self.chaos_iterations = 1000
        self.chaos_sequence = self.generate_chaotic_sequence(self.chaos_iterations)
        self.dim_mutation_prob = 0.2 # Probability for dimensional mutation
        self.adaptive_population_steps = 5

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

    def adaptive_dimensional_mutation(self, positions, lb, ub):
        mutation_indices = np.random.rand(positions.shape[0], self.dim) < self.dim_mutation_prob
        mutation_values = np.random.uniform(lb, ub, (positions.shape[0], self.dim))
        positions[mutation_indices] = mutation_values[mutation_indices]
        return positions

    def dynamic_population_resize(self, evaluations):
        factor = np.sin(evaluations / self.budget * np.pi)  # Oscillate over budget usage
        new_population_size = int(self.min_population_size + (self.max_population_size - self.min_population_size) * factor)
        return new_population_size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        positions = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(lb, ub, (population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        evaluations = population_size
        chaos_idx = 0
        
        while evaluations < self.budget:
            # Dynamic population resizing
            if evaluations % self.adaptive_population_steps == 0:
                population_size = self.dynamic_population_resize(evaluations)
                # Resize positions and velocities if needed
                if population_size < positions.shape[0]:
                    positions = positions[:population_size]
                    velocities = velocities[:population_size]
                    personal_best_positions = personal_best_positions[:population_size]
                    personal_best_scores = personal_best_scores[:population_size]
                else:
                    new_positions = np.random.uniform(lb, ub, (population_size - positions.shape[0], self.dim))
                    new_velocities = np.random.uniform(lb, ub, (population_size - velocities.shape[0], self.dim))
                    positions = np.vstack((positions, new_positions))
                    velocities = np.vstack((velocities, new_velocities))
                    new_scores = np.array([func(x) for x in new_positions])
                    personal_best_scores = np.hstack((personal_best_scores, new_scores))
                    personal_best_positions = np.vstack((personal_best_positions, new_positions))
                    evaluations += new_positions.shape[0]

            # Adaptive inertia weight reduction
            self.w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
            # Adaptive mutation factor reduction
            self.F = self.F_max - ((self.F_max - self.F_min) * (evaluations / self.budget))

            # Apply chaos to cognitive and social components
            chaos_c1 = self.c1 * self.chaos_sequence[chaos_idx % self.chaos_iterations]
            chaos_c2 = self.c2 * self.chaos_sequence[(chaos_idx + self.chaos_iterations // 2) % self.chaos_iterations]
            chaos_idx += 1

            r1, r2 = np.random.rand(population_size, self.dim), np.random.rand(population_size, self.dim)
            velocities = (self.w * velocities +
                          chaos_c1 * r1 * (personal_best_positions - positions) +
                          chaos_c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            positions = self.adaptive_dimensional_mutation(positions, lb, ub)

            if evaluations % 100 == 0:  # Enhanced adaptive boundary reset logic
                reset_scale = self.w * (ub - lb) * np.random.rand(population_size, self.dim)
                positions = np.random.uniform(lb, ub, (population_size, self.dim)) + reset_scale

            scores = np.array([func(x) for x in positions])
            evaluations += population_size

            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = positions[improved]
            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

            for i in range(population_size):
                if np.random.rand() < self.CR * (1 - evaluations / self.budget):  # Dynamic CR adjustment
                    indices = np.random.choice(population_size, 3, replace=False)
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