import numpy as np

class EnhancedAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.initial_velocity_scale = 0.1
        self.final_velocity_scale = 0.5
        self.covariance_inflation = 1e-3
        self.local_search_probability = 0.3  # Increased from 0.25 for more frequent local search

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * self.initial_velocity_scale
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])
        global_best_score = personal_best_scores[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                velocity_scale = self.initial_velocity_scale + (self.final_velocity_scale - self.initial_velocity_scale) * (evaluations / self.budget)
                population[i] += velocities[i] * velocity_scale

                # Apply a small probability of large mutation to escape local optima
                if np.random.rand() < 0.15:
                    cov_matrix = np.cov(population, rowvar=False) + self.covariance_inflation * np.eye(self.dim)
                    mutation_vector = np.random.multivariate_normal(np.zeros(self.dim), cov_matrix)
                    population[i] += mutation_vector

                population[i] = np.clip(population[i], lb, ub)

                score = func(population[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

            # Diversified local search with adaptive mutation step
            if np.random.rand() < self.local_search_probability:
                local_search_indices = np.random.choice(range(self.population_size), size=int(self.population_size * 0.2), replace=False)
                for idx in local_search_indices:
                    scale = np.random.rand() * 0.2 + 0.05  # Adaptive mutation scale
                    local_mutation_vector = np.random.normal(0, scale, size=self.dim)
                    population[idx] += local_mutation_vector
                    population[idx] = np.clip(population[idx], lb, ub)
                    local_score = func(population[idx])
                    evaluations += 1

                    if local_score < personal_best_scores[idx]:
                        personal_best_scores[idx] = local_score
                        personal_best_positions[idx] = population[idx]
                    if local_score < global_best_score:
                        global_best_score = local_score
                        global_best_position = population[idx]

        return global_best_score, global_best_position