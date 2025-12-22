import numpy as np

class EnhancedAdaptiveHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(5, 10 * dim)
        self.min_population_size = max(5, dim)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.decay_rate = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        eval_count = population_size

        def update_diversity():
            mean_position = np.mean(population, axis=0)
            diversity = np.mean(np.linalg.norm(population - mean_position, axis=1))
            return diversity

        while eval_count < self.budget:
            diversity = update_diversity()
            adaptive_lr = 0.1 + 0.4 * np.exp(-diversity)

            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                idxs = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[idxs]
                neighbor_idx = np.random.choice(population_size)
                neighbor_best = personal_best_positions[neighbor_idx]
                mutant_vector = a + self.mutation_factor * (b - c)
                adaptive_mutation_factor = 0.5 + 0.3 * np.abs(np.sin(0.1 * eval_count + np.pi)) * adaptive_lr
                mutant_vector = neighbor_best + adaptive_mutation_factor * (mutant_vector - neighbor_best)
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

            r1, r2 = np.random.rand(2, population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            velocities = np.clip(velocities, lb - population, ub - population)
            population += velocities
            population = np.clip(population, lb, ub)

            self.inertia_weight *= self.decay_rate
            self.mutation_factor = 0.5 + 0.3 * np.sin(0.1 * eval_count + 4 * np.random.random())
            self.mutation_factor *= 3.9 * self.mutation_factor * (1 - self.mutation_factor)
            self.crossover_prob = 0.5 + 0.4 * np.abs(np.sin(0.1 * eval_count + 0.5) + np.cos(0.05 * eval_count))
            self.cognitive_coeff = 1.5 + 0.5 * np.sin(0.01 * eval_count)
            self.social_coeff = 1.5 + 0.5 * np.cos(0.01 * eval_count)

            # Dynamic population adjustment based on diversity
            if diversity < 0.1:
                population_size = max(self.min_population_size, population_size - 1)
            elif diversity > 0.5:
                population_size = min(self.initial_population_size, population_size + 1)

            # Resize population and velocities
            population = np.resize(population, (population_size, self.dim))
            velocities = np.resize(velocities, (population_size, self.dim))
            personal_best_positions = np.resize(personal_best_positions, (population_size, self.dim))
            personal_best_scores = np.resize(personal_best_scores, population_size)

        return global_best_position