import numpy as np

class HybridEvoPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.mutation_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        eval_count = self.population_size

        while eval_count < self.budget:
            for i, position in enumerate(population):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - position)
                    + self.c2 * r2 * (global_best_position - position)
                )

                # Update position
                proposed_position = position + velocities[i]
                proposed_position = np.clip(proposed_position, lb, ub)

                # Mutation for diversity
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.uniform(lb, ub, self.dim)
                    proposed_position = self._hybrid_combine(proposed_position, mutation_vector)

                # Evaluate new position
                score = func(proposed_position)
                eval_count += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = proposed_position

                if score < func(global_best_position):
                    global_best_position = proposed_position

                if eval_count >= self.budget:
                    break

        return global_best_position

    def _hybrid_combine(self, pos, mut):
        return pos + (mut - pos) * np.random.rand()
