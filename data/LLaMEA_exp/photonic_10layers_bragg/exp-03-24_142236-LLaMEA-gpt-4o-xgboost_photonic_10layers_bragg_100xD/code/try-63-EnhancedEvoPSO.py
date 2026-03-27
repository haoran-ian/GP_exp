import numpy as np

class EnhancedEvoPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_population_size = self.population_size
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_rate = 0.1
        self.elite_count = 5
        self.diversity_threshold = 0.1
        self.learning_rate = 0.01  # Adaptive learning rate for personal best updating

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        eval_count = self.population_size

        while eval_count < self.budget:
            for i, position in enumerate(population):
                phase = eval_count / self.budget
                exploration_factor = self._calc_exploration_factor(phase)
                dynamic_c1, dynamic_c2, dynamic_w = self._calc_dynamic_parameters(phase, exploration_factor)

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    dynamic_w * velocities[i]
                    + dynamic_c1 * r1 * (personal_best_positions[i] - position)
                    + dynamic_c2 * r2 * (global_best_position - position)
                )

                proposed_position = position + velocities[i]
                proposed_position = np.clip(proposed_position, lb, ub)

                diversity = np.std(population, axis=0).mean()
                adaptive_mutation_rate = self.mutation_rate * (1 + diversity)
                if np.random.rand() < adaptive_mutation_rate:
                    elite_indices = np.argsort(personal_best_scores)[:self.elite_count]
                    elite_pos = personal_best_positions[elite_indices]
                    elite_vector = np.mean(elite_pos, axis=0)
                    mutation_vector = np.random.uniform(lb, ub, self.dim)
                    if np.linalg.norm(mutation_vector - position) > self.diversity_threshold * (ub - lb).mean():
                        proposed_position = self._adaptive_combine_diff(proposed_position, mutation_vector, elite_vector)

                score = func(proposed_position)
                eval_count += 1

                # Adaptive learning for personal best update
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = (1 - self.learning_rate) * personal_best_positions[i] + self.learning_rate * proposed_position

                if score < global_best_score:
                    global_best_position = proposed_position
                    global_best_score = score

                if eval_count >= self.budget:
                    break

            self.population_size = max(10, int(self.initial_population_size * (1 - phase)))
            if self.population_size < len(population):
                population = population[:self.population_size]
                velocities = velocities[:self.population_size]
                personal_best_positions = personal_best_positions[:self.population_size]
                personal_best_scores = personal_best_scores[:self.population_size]

        return global_best_position

    def _calc_exploration_factor(self, phase):
        if phase < 0.3:
            return 1 - 2 * phase
        elif phase < 0.6:
            return 0.5 * (1 - phase)
        else:
            return 0.1 * (1 - phase)

    def _calc_dynamic_parameters(self, phase, exploration_factor):
        dynamic_c1 = self.c1 * exploration_factor
        dynamic_c2 = self.c2 * (1 - exploration_factor)
        dynamic_w = self.w * (0.5 + 0.5 * np.random.rand()) * (0.9 - phase)
        return dynamic_c1, dynamic_c2, dynamic_w

    def _adaptive_combine_diff(self, pos, mut, elite):
        F = 0.8
        return pos + F * (elite - pos) + (mut - pos) * np.random.rand() ** 2