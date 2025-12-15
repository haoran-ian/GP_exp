import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 10 * dim)
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.inertia_decay = 0.98

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

            self.inertia_weight = 0.4 + (0.5 - 0.4) * np.exp(-0.01 * eval_count)
            self.mutation_factor = 0.5 + 0.3 * np.sin(0.1 * eval_count + np.pi/4)
            self.crossover_prob = 0.7 + 0.3 * np.cos(0.1 * eval_count + np.pi/4)
            self.cognitive_coeff = 2.0 + 0.5 * np.sin(0.01 * eval_count)
            self.social_coeff = 2.0 + 0.5 * np.cos(0.01 * eval_count)

        return global_best_position