import numpy as np

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 2)
        self.inertia_weight = 0.9  # Start with higher inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.max_temp = 100.0
        self.min_temp = 1.0
        self.temperature = self.max_temp
        self.cooling_rate = 0.95

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        search_space = bounds[1] - bounds[0]

        positions = np.random.rand(self.population_size, self.dim) * search_space + bounds[0]
        velocities = np.random.randn(self.population_size, self.dim) * 0.1 * search_space
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            # Rank-based selection for best positions
            scores = np.array([func(pos) for pos in positions])
            sorted_indices = np.argsort(scores)
            top_indices = sorted_indices[:self.population_size // 2]
            bottom_indices = sorted_indices[self.population_size // 2:]

            for i in range(self.population_size):
                # Adaptive inertia weight based on rank
                rank = np.where(sorted_indices == i)[0][0] / self.population_size
                dynamic_inertia_weight = self.inertia_weight * (1 - rank)

                inertia = dynamic_inertia_weight * velocities[i]
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia + cognitive_component + social_component

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])

                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score

                if score < global_best_score:
                    global_best_position = positions[i]
                    global_best_score = score

                if i in top_indices:
                    self.temperature = max(self.min_temp, self.temperature * self.cooling_rate)
                else:
                    self.temperature = self.max_temp  # Reset temperature for exploration

            # Rank-based velocity modification for added diversity
            for i in bottom_indices:
                velocities[i] *= np.random.rand() * 2

        return global_best_position, global_best_score