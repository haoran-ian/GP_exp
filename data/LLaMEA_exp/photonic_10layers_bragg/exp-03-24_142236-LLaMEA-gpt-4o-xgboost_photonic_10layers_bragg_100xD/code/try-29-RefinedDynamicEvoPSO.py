import numpy as np

class RefinedDynamicEvoPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.mutation_rate = 0.1
        self.elite_count = 5  # Number of elite members

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
            phase = eval_count / self.budget
            dynamic_w = self.w * (0.5 + 0.5 * np.random.rand()) * (1 - phase)

            for i, position in enumerate(population):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    dynamic_w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - position)
                    + self.c2 * r2 * (global_best_position - position)
                )
                proposed_position = position + velocities[i]
                proposed_position = np.clip(proposed_position, lb, ub)

                # Apply mutation if diversity is low
                if np.std(population, axis=0).mean() < 0.1:
                    if np.random.rand() < self.mutation_rate:
                        mutation_vector = np.random.uniform(lb, ub, self.dim)
                        proposed_position = self._hybrid_combine(proposed_position, mutation_vector)

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

            # Dynamic population resizing
            self.population_size = max(10, int(self.initial_population_size * (1 - phase)))
            if self.population_size < len(population):
                population = population[:self.population_size]
                velocities = velocities[:self.population_size]
                personal_best_positions = personal_best_positions[:self.population_size]
                personal_best_scores = personal_best_scores[:self.population_size]

        return global_best_position

    def _hybrid_combine(self, pos, mut):
        return pos + (mut - pos) * np.random.rand()