import numpy as np

class EnhancedDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, 10 * dim)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.decay_rate = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        eval_count = self.population_size

        def update_diversity():
            mean_position = np.mean(population, axis=0)
            diversity = np.mean(np.linalg.norm(population - mean_position, axis=1))
            return diversity

        while eval_count < self.budget:
            diversity = update_diversity()
            adaptive_lr = 0.3 + 0.2 * np.sin(eval_count / self.budget)

            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant_vector = a + self.mutation_factor * (b - c)

                exploration_exploitation_balance = 0.5 * (1 + np.tanh(5 * (0.5 - eval_count / self.budget)))
                adaptive_mutation_factor = 0.6 + 0.2 * np.cos(eval_count / self.budget)

                if exploration_exploitation_balance > 0.5:
                    mutant_vector = (1 - adaptive_mutation_factor) * mutant_vector + adaptive_mutation_factor * np.mean(population, axis=0)
                else:
                    mutant_vector = (1 - adaptive_mutation_factor) * mutant_vector + adaptive_mutation_factor * global_best_position

                mutant_vector = np.clip(mutant_vector, lb, ub)
                trial_vector = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < (self.crossover_prob + 0.1 * diversity)
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
            velocities *= (0.5 + 0.5 * np.tanh(diversity))
            velocities = np.clip(velocities, lb - population, ub - population)
            population += velocities
            population = np.clip(population, lb, ub)

            # Update strategies
            self.inertia_weight *= (self.decay_rate - 0.3 * np.tanh(diversity))  # Adaptive inertia damping
            self.mutation_factor = 0.5 + 0.5 * np.sin(eval_count / self.budget)
            self.crossover_prob = 0.5 + 0.4 * np.abs(np.sin(0.1 * eval_count) + np.cos(0.05 * eval_count))
            self.cognitive_coeff = 1.5 + 0.5 * np.sin(0.01 * eval_count)
            self.social_coeff = 1.5 + 0.5 * np.cos(0.01 * eval_count)

            # Implement adaptive multi-phase strategy
            if eval_count / self.budget < 0.3:
                self.inertia_weight = max(0.4, self.inertia_weight * 1.05)
                self.cognitive_coeff = min(2.0, self.cognitive_coeff * 1.1)
            elif eval_count / self.budget < 0.7:
                self.social_coeff = max(1.5, self.social_coeff * 0.9)
            else:
                self.mutation_factor = min(1.0, self.mutation_factor * 1.1)
                self.crossover_prob = max(0.6, self.crossover_prob * 0.95)

        return global_best_position