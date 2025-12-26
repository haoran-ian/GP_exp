import numpy as np

class RefinedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(60, self.budget // 4)
        self.inertia_weight = 0.4
        self.cognitive_coeff = 1.2
        self.social_coeff = 1.8
        self.initial_learning_factor = 0.8
        self.final_learning_factor = 0.3
        self.initial_temperature = 150.0
        self.cooling_rate = 0.996
        self.min_population_size = 8
        self.differential_weight = 0.7
        self.crossover_rate = 0.9

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

            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = personal_best_positions[a] + self.differential_weight * (personal_best_positions[b] - personal_best_positions[c])
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.crossover_rate else positions[i, j] for j in range(self.dim)])
                trial_vector = np.clip(trial_vector, bounds[0], bounds[1])
                
                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                if trial_score < global_best_score:
                    global_best_position = trial_vector
                    global_best_score = trial_score

                inertia = self.inertia_weight * velocities[i] * (1 - evaluations / self.budget)
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia + cognitive_component + social_component
                velocities[i] *= learning_factor

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])

                if np.random.rand() < np.exp(-abs(trial_score - global_best_score) / temperature):
                    velocities[i] *= np.random.rand() * 2

            if evaluations / self.budget > 0.5:
                population_size = max(self.min_population_size, population_size - 1)
                positions = positions[:population_size]
                velocities = velocities[:population_size]
                personal_best_positions = personal_best_positions[:population_size]
                personal_best_scores = personal_best_scores[:population_size]

            temperature *= self.cooling_rate

        return global_best_position, global_best_score