import numpy as np

class ImprovedAdaptiveParticleSwarmLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.w_min = 0.2
        self.w_max = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_scale = 0.1
        self.local_search_probability = 0.2
        self.local_search_radius = 0.1

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
            # Adaptive inertia weight
            self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

            for i in range(self.population_size):
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                population[i] += velocities[i] * (self.velocity_scale * 0.5 + 0.5)
                population[i] = np.clip(population[i], lb, ub)

                score = func(population[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

            if np.random.rand() < self.local_search_probability:
                local_search_indices = np.random.choice(range(self.population_size), size=int(self.population_size * 0.2), replace=False)
                for idx in local_search_indices:
                    # Dynamic local search radius
                    dynamic_radius = self.local_search_radius * (1 - evaluations / self.budget)
                    local_mutation_vector = np.random.normal(0, dynamic_radius, size=self.dim)
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