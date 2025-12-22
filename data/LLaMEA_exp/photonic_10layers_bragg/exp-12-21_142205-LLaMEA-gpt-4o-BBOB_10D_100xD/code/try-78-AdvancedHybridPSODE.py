import numpy as np

class AdvancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, self.budget // 5)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 2.0
        self.social_coeff = 1.7
        self.mutation_factor = 0.9
        self.crossover_rate = 0.95
        self.adaptive_mutation_factor = 0.5 + np.random.rand() / 2
        self.dynamic_scaling_factor = 0.1 + np.random.rand() / 10

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        eval_count = self.population_size

        while eval_count < self.budget:
            # PSO Component with Adaptive Parameters
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            population = np.clip(population + velocities, lb, ub)

            # DE Component with Adaptive Mutation
            for i in range(self.population_size):
                candidates = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[candidates]
                mutant = np.clip(x1 + self.adaptive_mutation_factor * (x2 - x3), lb, ub)
                
                # Adaptive Crossover
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

            # Dynamic Population Resizing
            if eval_count < self.budget / 2 and self.population_size < int(self.budget / 10):
                self.population_size += int(self.dynamic_scaling_factor * self.population_size)
                velocities = np.concatenate(
                    (velocities, np.zeros((self.population_size - velocities.shape[0], self.dim))), axis=0)
                personal_best_positions = np.concatenate(
                    (personal_best_positions, np.random.uniform(lb, ub, (self.population_size - personal_best_positions.shape[0], self.dim))), axis=0)
                personal_best_scores = np.concatenate(
                    (personal_best_scores, np.array([func(ind) for ind in personal_best_positions[-(self.population_size - len(personal_best_scores)):]])), axis=0)
        
        return global_best_position, global_best_score