import numpy as np

class RefinedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = min(100, self.budget // 5)
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 2.0
        self.social_coeff = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.base_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        eval_count = population_size

        while eval_count < self.budget:
            # Dynamic Inertia Weight
            inertia_weight = (self.inertia_weight_max - self.inertia_weight_min) * \
                             (self.budget - eval_count) / self.budget + self.inertia_weight_min

            # PSO Component
            r1, r2 = np.random.rand(2, population_size, self.dim)
            velocities = (inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            population = np.clip(population + velocities, lb, ub)

            # DE Component
            for i in range(population_size):
                candidates = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[candidates]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3), lb, ub)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])
                trial_score = func(trial)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score

                    if trial_score < global_best_score:
                        global_best_position = trial
                        global_best_score = trial_score

                if eval_count >= self.budget:
                    break

            # Dynamic Population Size Adjustment
            if eval_count < self.budget / 2:
                population_size = min(self.base_population_size + eval_count // 10, self.budget // 2)
                population = population[:population_size]
                velocities = velocities[:population_size]
                personal_best_positions = personal_best_positions[:population_size]
                personal_best_scores = personal_best_scores[:population_size]

        return global_best_position, global_best_score