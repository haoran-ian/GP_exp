import numpy as np

class EnhancedAdaptiveParticleSwarmILS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_scale = 0.1
        self.covariance_inflation = 1e-3
        self.local_search_probability = 0.2
        self.success_count = 0
        self.failure_count = 0

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
            inertia_weight = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            cov_matrix = np.cov(population, rowvar=False) + self.covariance_inflation * np.eye(self.dim)

            for i in range(self.population_size):
                velocities[i] = (
                    inertia_weight * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                population[i] += velocities[i] * self.velocity_scale

                if np.random.rand() < 0.1:
                    mutation_vector = np.random.multivariate_normal(np.zeros(self.dim), cov_matrix)
                    population[i] += mutation_vector

                population[i] = np.clip(population[i], lb, ub)

                score = func(population[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                    self.success_count += 1
                    self.failure_count = 0
                elif np.random.rand() < 0.05:  # add stochastic acceptance for exploration
                    personal_best_scores[i] *= 1.01  # artificially inflate score for variety
                    self.failure_count += 1

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

                # Adjust the velocity scale based on the success history
                if self.success_count > self.failure_count:
                    self.velocity_scale = min(0.5, self.velocity_scale * 1.1)
                else:
                    self.velocity_scale *= 0.9

            # Dynamic local search probability based on success rate
            if self.success_count > 5:
                if np.random.rand() < self.local_search_probability:
                    local_search_indices = np.random.choice(range(self.population_size), size=int(self.population_size * 0.2), replace=False)
                    for idx in local_search_indices:
                        local_mutation_vector = np.random.normal(0, 0.1, size=self.dim)
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

                self.success_count = 0

        return global_best_score, global_best_position