import numpy as np

class EnhancedVelocitySwarmMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Heuristic population size
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.mutation_rate = 0.1  # Initial mutation rate
        self.velocity_scale = 0.1  # Initial velocity scale factor
        self.neighborhood_size = int(0.1 * self.population_size)  # Neighborhood size for local search

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
            for i in range(self.population_size):
                # Select neighborhood
                neighbors_indices = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
                local_best_index = neighbors_indices[np.argmin(personal_best_scores[neighbors_indices])]
                local_best_position = personal_best_positions[local_best_index]

                # Velocity update with neighborhood influence
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (local_best_position - population[i])
                )
                population[i] += velocities[i] * self.velocity_scale

                # Mutation step: Adaptive mutation rate based on proximity to global and local best
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

                # Adjust mutation rate and velocity scale dynamically
                diversity = np.mean(np.linalg.norm(population - global_best_position, axis=1))
                self.mutation_rate = 0.05 + 0.45 * (1 - evaluations / self.budget) * (diversity / np.linalg.norm(ub - lb))
                self.velocity_scale = 0.1 + 0.9 * (evaluations / self.budget)

        return global_best_score, global_best_position