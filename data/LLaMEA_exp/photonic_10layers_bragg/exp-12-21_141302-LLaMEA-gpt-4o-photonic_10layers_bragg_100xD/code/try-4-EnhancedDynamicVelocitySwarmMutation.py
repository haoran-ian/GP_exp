import numpy as np

class EnhancedDynamicVelocitySwarmMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Heuristic population size
        self.base_inertia = 0.9  # Base inertia weight
        self.min_inertia = 0.4  # Minimum inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.mutation_rate = 0.1  # Initial mutation rate
        self.velocity_scale = 0.1  # Initial velocity scale factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * self.velocity_scale
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])
        global_best_score = personal_best_scores[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            diversity = np.mean(np.linalg.norm(population - global_best_position, axis=1))
            inertia_weight = self.min_inertia + (self.base_inertia - self.min_inertia) * (diversity / np.linalg.norm(ub - lb))

            for i in range(self.population_size):
                velocities[i] = (
                    inertia_weight * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                # Momentum preservation: retain a fraction of previous velocity
                velocities[i] = 0.9 * velocities[i] + 0.1 * np.random.uniform(-1, 1, self.dim)
                population[i] += velocities[i] * self.velocity_scale

                # Mutation step: Adaptive mutation rate based on proximity to global best
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(0, (ub - lb) * 0.1, self.dim)
                    population[i] += mutation_vector

                # Ensure within bounds
                population[i] = np.clip(population[i], lb, ub)

                # Evaluate new position
                score = func(population[i])
                evaluations += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

                # Adjust mutation rate dynamically
                self.mutation_rate = 0.05 + 0.45 * (1 - evaluations / self.budget) * (diversity / np.linalg.norm(ub - lb))

        return global_best_score, global_best_position