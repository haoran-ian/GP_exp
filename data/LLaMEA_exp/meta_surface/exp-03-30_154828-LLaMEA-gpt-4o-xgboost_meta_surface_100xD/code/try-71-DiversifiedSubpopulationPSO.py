import numpy as np

class DiversifiedSubpopulationPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.num_subpopulations = 5
        self.subpopulation_size = self.population_size // self.num_subpopulations
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.mutation_rate = 0.2
        self.chaos_iterations = 1000
        self.chaos_sequence = self.generate_chaotic_sequence(self.chaos_iterations)

    def generate_chaotic_sequence(self, length):
        x = 0.7
        r = 3.9
        sequence = []
        for _ in range(length):
            x = r * x * (1 - x)
            sequence.append(x)
        return sequence

    def levy_flight(self, step_scale=0.03):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 1.5))
        return step_scale * step

    def dynamic_mutation(self, positions, lb, ub, evaluations):
        mutation_prob = self.mutation_rate * (1 - evaluations / self.budget)
        mutation_indices = np.random.rand(self.population_size, self.dim) < mutation_prob
        mutation_values = np.random.uniform(lb, ub, (self.population_size, self.dim))
        positions[mutation_indices] = mutation_values[mutation_indices]
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
            w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
            chaos_c1 = self.c1 * self.chaos_sequence[(chaos_idx + evaluations) % self.chaos_iterations]
            chaos_c2 = self.c2 * self.chaos_sequence[(chaos_idx + self.chaos_iterations // 2) % self.chaos_iterations]
            chaos_idx += 1

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          chaos_c1 * r1 * (personal_best_positions - positions) +
                          chaos_c2 * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            positions = self.dynamic_mutation(positions, lb, ub, evaluations)

            scores = np.array([func(x) for x in positions])
            evaluations += self.population_size

            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = positions[improved]
            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

            # Diversify subpopulations
            for sub in range(self.num_subpopulations):
                if evaluations % (self.budget // self.num_subpopulations) == 0:
                    start = sub * self.subpopulation_size
                    end = (sub + 1) * self.subpopulation_size
                    sub_positions = positions[start:end]
                    if np.random.rand() < 0.3:  # Add more frequent diversification
                        sub_positions = np.random.uniform(lb, ub, (self.subpopulation_size, self.dim))
                    scores[start:end] = [func(x) for x in sub_positions]
                    evaluations += self.subpopulation_size
                    positions[start:end] = sub_positions

        return global_best_position, personal_best_scores[global_best_index]