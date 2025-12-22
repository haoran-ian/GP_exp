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
        self.global_learning_rate = 0.1
        self.personal_learning_rate = 0.1
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

                # Update velocities with agent-specific learning rates
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.personal_learning_rate * r1 * (personal_best_positions[i] - population[i]) +
                                 self.global_learning_rate * r2 * (global_best_position - population[i]))
                velocities[i] = np.clip(velocities[i], lb - population[i], ub - population[i])
                population[i] += velocities[i]
                population[i] = np.clip(population[i], lb, ub)

            self.inertia_weight *= self.decay_rate
            self.mutation_factor = 0.9 - 0.5 * np.sin(0.1 * eval_count)  # Adaptive mutation adjustment
            self.crossover_prob *= (1.0 - 0.01 * eval_count / self.budget)  # Adaptive crossover probability
            self.global_learning_rate = 0.1 + 0.1 * np.sin(0.05 * eval_count)  # Modify global influence dynamically
            self.personal_learning_rate = 0.1 + 0.1 * np.cos(0.05 * eval_count)  # Modify personal influence dynamically

        return global_best_position