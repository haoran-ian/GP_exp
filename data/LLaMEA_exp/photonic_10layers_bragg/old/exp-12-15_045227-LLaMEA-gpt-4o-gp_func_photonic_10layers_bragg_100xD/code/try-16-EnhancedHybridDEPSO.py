import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 10 * dim)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.decay_rate = 0.99
        self.chaotic_sequence = np.random.rand(self.population_size)
        self.neighborhood_radius = dim * 0.1  # Adaptive neighborhood radius

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

            # Chaotic search phase
            self.chaotic_sequence = 4 * self.chaotic_sequence * (1 - self.chaotic_sequence)
            chaotic_indexes = (self.chaotic_sequence * self.population_size).astype(int)
            chaotic_positions = population[chaotic_indexes]
            chaotic_scores = np.array([func(ind) for ind in chaotic_positions])
            eval_count += self.population_size

            for idx, score in enumerate(chaotic_scores):
                if score < personal_best_scores[chaotic_indexes[idx]]:
                    personal_best_scores[chaotic_indexes[idx]] = score
                    personal_best_positions[chaotic_indexes[idx]] = chaotic_positions[idx]

            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            velocities = np.clip(velocities, lb - population, ub - population)
            population += velocities
            population = np.clip(population, lb, ub)

            # Adaptive neighborhood influence
            for i in range(self.population_size):
                neighbors = np.linalg.norm(population - population[i], axis=1) < self.neighborhood_radius
                neighborhood_best_idx = np.argmin(personal_best_scores[neighbors])
                if personal_best_scores[neighbors][neighborhood_best_idx] < personal_best_scores[i]:
                    population[i] = personal_best_positions[neighbors][neighborhood_best_idx]

            self.inertia_weight *= self.decay_rate
            self.mutation_factor = 0.5 + 0.3 * np.sin(0.1 * eval_count + 4 * np.random.random())
            self.mutation_factor *= 3.9 * self.mutation_factor * (1 - self.mutation_factor)
            self.crossover_prob = 0.5 + 0.4 * np.abs(np.sin(0.1 * eval_count + 0.5))
            self.cognitive_coeff = 1.5 + 0.5 * np.sin(0.01 * eval_count)
            self.social_coeff = 1.5 + 0.5 * np.cos(0.01 * eval_count)

        return global_best_position