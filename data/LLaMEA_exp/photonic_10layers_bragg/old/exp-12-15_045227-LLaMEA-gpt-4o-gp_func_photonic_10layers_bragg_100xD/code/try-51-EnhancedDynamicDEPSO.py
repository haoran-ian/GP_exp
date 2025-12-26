import numpy as np

class EnhancedDynamicDEPSO:
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
        self.exploration_exploitation_ratio = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        eval_count = self.population_size

        def levy_flight(Lambda):
            sigma = (np.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
                     (np.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
            u = np.random.randn(self.dim) * sigma
            v = np.random.randn(self.dim)
            step = u / np.abs(v) ** (1 / Lambda)
            return step

        while eval_count < self.budget:
            diversity = np.mean(np.linalg.norm(population - np.mean(population, axis=0), axis=1))
            adaptive_lr = 0.1 + 0.4 * np.exp(-diversity)
            phase_switch = eval_count / self.budget
            is_exploration_phase = phase_switch < self.exploration_exploitation_ratio

            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                mutant_vector = a + self.mutation_factor * (b - c)
                chaotic_factor = np.sin(0.9 * (i + eval_count))
                adaptive_mutation_factor = 0.5 + 0.3 * chaotic_factor * adaptive_lr

                if is_exploration_phase:
                    mutant_vector = (1 - adaptive_mutation_factor) * mutant_vector + adaptive_mutation_factor * np.mean(population, axis=0) + 0.01 * levy_flight(1.5)
                else:
                    mutant_vector = (1 - adaptive_mutation_factor) * mutant_vector + adaptive_mutation_factor * global_best_position

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
            velocities *= 0.5
            velocities = np.clip(velocities, lb - population, ub - population)
            population += velocities
            population = np.clip(population, lb, ub)

            self.inertia_weight *= 0.9
            self.mutation_factor = 0.8 + 0.3 * np.sin(0.1 * eval_count + np.random.random())
            self.mutation_factor *= 3.9 * self.mutation_factor * (1 - self.mutation_factor)
            chaotic_factor += 0.1 * np.sin(0.05 * eval_count)
            self.crossover_prob = 0.7 + 0.4 * np.abs(np.sin(0.1 * eval_count + np.pi) + np.cos(0.05 * eval_count))
            self.cognitive_coeff = 1.5 + 0.5 * np.sin(0.01 * eval_count) * chaotic_factor
            self.social_coeff = 1.5 + 0.5 * np.cos(0.01 * eval_count)

        return global_best_position