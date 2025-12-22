import numpy as np

class EnhancedAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.vel_limit = 0.2
        self.local_search_probability = 0.3

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * self.vel_limit
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])
        global_best_score = personal_best_scores[global_best_index]

        evaluations = self.population_size

        def calculate_diversity(pop):
            mean_position = np.mean(pop, axis=0)
            return np.mean(np.linalg.norm(pop - mean_position, axis=1))

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            diversity = calculate_diversity(population)

            for i in range(self.population_size):
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i])
                    + self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                )
                velocities[i] = np.clip(velocities[i], -self.vel_limit, self.vel_limit)
                population[i] += velocities[i]

                mutation_strength = 0.1 / (1 + np.exp(-diversity + 2))
                if np.random.rand() < mutation_strength:
                    mutation_vector = np.random.normal(0, mutation_strength, self.dim)
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

            if np.random.rand() < self.local_search_probability:
                local_search_indices = np.random.choice(range(self.population_size), size=int(self.population_size * 0.2), replace=False)
                for idx in local_search_indices:
                    local_mutation_vector = np.random.normal(0, 0.05, size=self.dim)
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