import numpy as np

class AdvancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(60, self.budget // 4)
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.initial_learning_factor = 0.9
        self.final_learning_factor = 0.3
        self.initial_temperature = 100.0
        self.cooling_rate = 0.995
        self.min_population_size = 15
        self.differential_weight = 0.8
        self.crossover_rate = 0.7

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
                inertia = self.inertia_weight * velocities[i] * (1 - evaluations / self.budget)
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia + cognitive_component + social_component
                velocities[i] *= learning_factor

                # Differential mutation and crossover
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = positions[indices[0]], positions[indices[1]], positions[indices[2]]
                mutant_vector = a + self.differential_weight * (b - c)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, positions[i])
                trial_vector = np.clip(trial_vector, bounds[0], bounds[1])

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                if trial_score < global_best_score:
                    global_best_position = trial_vector
                    global_best_score = trial_score
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])

            temperature *= self.cooling_rate

            if evaluations / self.budget > 0.5:
                population_size = max(self.min_population_size, population_size - 1)
                positions = np.resize(positions, (population_size, self.dim))
                velocities = np.resize(velocities, (population_size, self.dim))
                personal_best_positions = np.resize(personal_best_positions, (population_size, self.dim))
                personal_best_scores = np.resize(personal_best_scores, population_size)

        return global_best_position, global_best_score