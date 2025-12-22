import numpy as np

class ImprovedAdaptiveParticleSwarmLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_scale = 0.1
        self.covariance_inflation = 1e-3
        self.local_search_probability = 0.2

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
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            cov_matrix = np.cov(population, rowvar=False) + self.covariance_inflation * np.eye(self.dim)

            for i in range(self.population_size):
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                population[i] += velocities[i]
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

            fitness_gain = (global_best_score - np.min(personal_best_scores)) / (global_best_score + 1e-10)
            local_search_prob = self.local_search_probability * (1 + fitness_gain)
            
            if np.random.rand() < local_search_prob:
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

        return global_best_score, global_best_position