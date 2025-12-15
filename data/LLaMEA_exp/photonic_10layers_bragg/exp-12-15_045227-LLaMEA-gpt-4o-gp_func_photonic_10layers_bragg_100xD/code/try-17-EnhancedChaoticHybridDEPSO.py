import numpy as np

class EnhancedChaoticHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 10 * dim)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.decay_rate = 0.99  # Adaptive parameter for inertia weight
        self.escape_velocity_factor = 1.2  # New parameter for escaping local optima

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant_vector = a + self.mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                trial_vector = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]
                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector

                if trial_score < personal_best_scores[global_best_idx]:
                    global_best_idx = i
                    global_best_position = trial_vector

            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            velocities = np.clip(velocities, lb - population, ub - population)
            population += velocities
            population = np.clip(population, lb, ub)

            # Chaotic adaptation for mutation and crossover
            self.mutation_factor = 0.5 + 0.3 * np.sin(0.1 * eval_count)
            self.mutation_factor *= self.escape_velocity_factor * (3.9 * self.mutation_factor * (1 - self.mutation_factor))
            self.crossover_prob = 0.5 + 0.4 * np.abs(np.sin(0.1 * eval_count + 0.5))

            # Adaptive learning rates
            self.cognitive_coeff = 1.5 + 0.5 * np.sin(0.01 * eval_count)
            self.social_coeff = 1.5 + 0.5 * np.cos(0.01 * eval_count)
            self.inertia_weight *= self.decay_rate

            # Escape mechanism for better exploration
            if np.random.rand() < 0.05:
                worst_idx = np.argmax(personal_best_scores)
                velocities[worst_idx] *= self.escape_velocity_factor
                population[worst_idx] = np.random.uniform(lb, ub, self.dim)

        return global_best_position