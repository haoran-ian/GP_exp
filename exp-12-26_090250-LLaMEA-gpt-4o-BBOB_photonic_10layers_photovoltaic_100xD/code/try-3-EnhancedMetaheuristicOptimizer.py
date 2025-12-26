import numpy as np

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 2)
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.differential_weight = 0.8
        self.crossover_prob = 0.9

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
            for i in range(self.population_size):
                rand_indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = positions[rand_indices]
                mutant_vector = a + self.differential_weight * (b - c)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, positions[i])

                trial_vector = np.clip(trial_vector, bounds[0], bounds[1])
                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                if trial_score < global_best_score:
                    global_best_position = trial_vector
                    global_best_score = trial_score

                inertia = self.inertia_weight * velocities[i]
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia + cognitive_component + social_component
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])

                if np.random.rand() < np.exp(-abs(trial_score - global_best_score) / self.temperature):
                    velocities[i] *= np.random.rand() * 2

            self.temperature *= self.cooling_rate

        return global_best_position, global_best_score