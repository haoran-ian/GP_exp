import numpy as np

class OptimizedDynamicEvoPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.initial_population_size = self.population_size
        self.w_min = 0.3
        self.w_max = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_rate = 0.1
        self.elite_count = 5
        self.reinitialize_frequency = 0.1  # Fraction of budget to trigger reinitialization

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        eval_count = self.population_size
        reinitialize_threshold = int(self.budget * self.reinitialize_frequency)

        while eval_count < self.budget:
            for i, position in enumerate(population):
                # Adjust inertia weight dynamically
                phase = eval_count / self.budget
                self.w = self.w_max - (self.w_max - self.w_min) * phase

                # Update velocity with adaptive coefficients
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - position)
                    + self.c2 * r2 * (global_best_position - position)
                )

                # Update position
                proposed_position = position + velocities[i]
                proposed_position = np.clip(proposed_position, lb, ub)

                # Elite-guided adaptive mutation
                elite_indices = np.argsort(personal_best_scores)[:self.elite_count]
                elite_pos = personal_best_positions[elite_indices]
                elite_vector = np.mean(elite_pos, axis=0)
                adaptive_mutation_rate = self.mutation_rate * (1 - phase)
                if np.random.rand() < adaptive_mutation_rate:
                    mutation_vector = np.random.uniform(lb, ub, self.dim)
                    proposed_position = self._hybrid_combine(proposed_position, mutation_vector, elite_vector)

                # Evaluate new position
                score = func(proposed_position)
                eval_count += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = proposed_position

                if score < global_best_score:
                    global_best_position = proposed_position
                    global_best_score = score

                if eval_count >= self.budget:
                    break

            # Dynamic population adjustment
            self.population_size = max(10, int(self.initial_population_size * (1 - phase)))
            if self.population_size < len(population):
                population = population[:self.population_size]
                velocities = velocities[:self.population_size]
                personal_best_positions = personal_best_positions[:self.population_size]
                personal_best_scores = personal_best_scores[:self.population_size]

            # Periodic reinitialization to escape local optima
            if eval_count % reinitialize_threshold == 0:
                for i in range(int(self.population_size * 0.1)):  # Reinitialize 10% of the population
                    population[i] = np.random.uniform(lb, ub, self.dim)
                    velocities[i] = np.random.uniform(-1, 1, self.dim)
                    personal_best_positions[i] = np.copy(population[i])
                    personal_best_scores[i] = func(population[i])
                    eval_count += 1
                    if eval_count >= self.budget:
                        break

        return global_best_position

    def _hybrid_combine(self, pos, mut, elite):
        return pos + (mut - pos) * np.random.rand() + (elite - pos) * np.random.rand()