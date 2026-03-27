import numpy as np

class EnhancedAdaptiveDynamicEvoPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_population_size = self.population_size
        self.w_min = 0.4  # minimum inertia weight
        self.w_max = 0.9  # maximum inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.mutation_rate = 0.1
        self.elite_count = 5
        self.neighborhood_size = 5

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
                exploration_factor = 1 - phase ** 2

                # Adaptive inertia weight
                dynamic_w = self.w_max - (self.w_max - self.w_min) * phase

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    dynamic_w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - position)
                    + self.c2 * r2 * (global_best_position - position)
                )

                proposed_position = position + velocities[i]
                proposed_position = np.clip(proposed_position, lb, ub)

                # Phase-based elitism
                if phase < 0.5:
                    local_best = global_best_position
                else:
                    elite_indices = np.argsort(personal_best_scores)[:self.elite_count]
                    elite_positions = personal_best_positions[elite_indices]
                    local_best = elite_positions[np.random.choice(elite_indices)]

                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.uniform(lb, ub, self.dim)
                    proposed_position = self._blend_positions(proposed_position, mutation_vector, local_best)

                score = func(proposed_position)
                eval_count += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = proposed_position

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

    def _blend_positions(self, pos, mut, elite):
        alpha = np.random.rand()
        return alpha * mut + (1 - alpha) * elite