import numpy as np

class EnhancedAdaptiveParticleSwarmDM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_scale = 0.1
        self.covariance_inflation = 1e-3
        self.local_search_probability = 0.2
        self.differential_weight = 0.8

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
            cov_matrix = np.cov(population, rowvar=False) + self.covariance_inflation * np.eye(self.dim)

            for i in range(self.population_size):
                velocities[i] = (
                    self.w * velocities[i]
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
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

                self.velocity_scale = 0.1 + 0.8 * (evaluations / self.budget)

            if np.random.rand() < self.local_search_probability:
                for idx in np.random.choice(range(self.population_size), size=int(self.population_size * 0.2), replace=False):
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    differential_mutation = a + self.differential_weight * (b - c)
                    differential_mutation = np.clip(differential_mutation, lb, ub)
                    
                    if func(differential_mutation) < personal_best_scores[idx]:
                        personal_best_scores[idx] = func(differential_mutation)
                        personal_best_positions[idx] = differential_mutation
                    if func(differential_mutation) < global_best_score:
                        global_best_score = func(differential_mutation)
                        global_best_position = differential_mutation
                    evaluations += 1

        return global_best_score, global_best_position