import numpy as np

class RefinedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(50, self.budget // 5)
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.0
        self.social_coeff = 1.0
        self.initial_temperature = 100.0
        self.cooling_rate = 0.993
        self.min_population_size = 10
        self.epsilon = 1e-8

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

        learning_rates = np.full(population_size, 0.5)
        success_counter = np.zeros(population_size)

        while evaluations < self.budget:
            for i in range(population_size):
                inertia = self.inertia_weight * velocities[i]
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                
                velocities[i] = inertia + cognitive_component + social_component
                velocities[i] *= learning_rates[i]

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])

                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score
                    success_counter[i] += 1
                else:
                    success_counter[i] = max(0, success_counter[i] - 1)

                if score < global_best_score:
                    global_best_position = positions[i]
                    global_best_score = score

                if np.random.rand() < np.exp(-abs(score - global_best_score) / (temperature + self.epsilon)):
                    velocities[i] *= np.random.rand() * 2

                learning_rates[i] = min(1.0, max(0.1, learning_rates[i] + 0.1 * (success_counter[i] - 1)))

            if evaluations / self.budget > 0.5:
                population_size = max(self.min_population_size, population_size - 1)
                positions = positions[:population_size]
                velocities = velocities[:population_size]
                personal_best_positions = personal_best_positions[:population_size]
                personal_best_scores = personal_best_scores[:population_size]
                learning_rates = learning_rates[:population_size]
                success_counter = success_counter[:population_size]

            temperature *= self.cooling_rate

        return global_best_position, global_best_score