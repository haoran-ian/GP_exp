import numpy as np

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(50, self.budget // 5)
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.initial_learning_factor = 0.9
        self.final_learning_factor = 0.4
        self.initial_temperature = 100.0
        self.cooling_rate = 0.995
        self.min_population_size = 10

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        search_space = bounds[1] - bounds[0]
        temperature = self.initial_temperature
        population_size = self.initial_population_size
        
        positions = np.random.rand(population_size, self.dim) * search_space + bounds[0]
        velocities = np.random.randn(population_size, self.dim) * 0.1 * search_space
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = population_size

        while evaluations < self.budget:
            learning_factor = (self.initial_learning_factor - self.final_learning_factor) * (1 - evaluations / self.budget) + self.final_learning_factor
            self.inertia_weight *= 0.99  # Inertia weight decay

            for i in range(population_size):
                inertia = self.inertia_weight * velocities[i] * (1 - evaluations / self.budget)
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia + cognitive_component + social_component
                velocities[i] *= learning_factor

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

                if np.random.rand() < np.exp(-abs(score - global_best_score) / temperature):
                    velocities[i] *= np.random.rand() * 2

            if evaluations / self.budget > 0.5:
                population_size = max(self.min_population_size, population_size - 1)
                positions = positions[:population_size]
                velocities = velocities[:population_size]
                personal_best_positions = personal_best_positions[:population_size]
                personal_best_scores = personal_best_scores[:population_size]

            temperature *= self.cooling_rate * (1 - evaluations / self.budget)  # Change line

        return global_best_position, global_best_score