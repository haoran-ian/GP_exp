import numpy as np

class AdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.CR = 0.9
        self.F = 0.8
        self.elite_fraction = 0.1
        self.pop_diversity_threshold = 0.1
        self.sub_population_sizes = [30, 20]
        self.populations = [np.random.uniform(-1, 1, (size, self.dim)) for size in self.sub_population_sizes]
        self.velocities = [np.random.uniform(-1, 1, (size, self.dim)) for size in self.sub_population_sizes]
        self.personal_best_positions = [np.copy(pop) for pop in self.populations]
        self.personal_best_scores = [np.full(size, np.inf) for size in self.sub_population_sizes]
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        evaluations = 0
        global_best_positions = []
        global_best_scores = []

        # Initialize personal best scores
        for pop_idx, (population, personal_best_scores) in enumerate(zip(self.populations, self.personal_best_scores)):
            for idx, individual in enumerate(population):
                score = func(individual)
                evaluations += 1
                personal_best_scores[idx] = score
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = individual

        while evaluations < self.budget:
            for pop_idx, (population, velocities, personal_best_scores, personal_best_positions) in enumerate(
                    zip(self.populations, self.velocities, self.personal_best_scores, self.personal_best_positions)):

                # Dynamic inertia weight based on diversity
                diversity = np.std(population, axis=0).mean()
                dynamic_inertia = self.inertia_weight * (1 - (diversity / self.pop_diversity_threshold))

                elite_count = max(1, int(self.elite_fraction * len(population)))
                sorted_indices = np.argsort(personal_best_scores)
                elites = population[sorted_indices[:elite_count]]

                # PSO update with adaptive parameters
                for i, individual in enumerate(population):
                    r1, r2 = np.random.rand(2)
                    velocities[i] = (
                        dynamic_inertia * velocities[i] +
                        self.cognitive_coeff * r1 * (personal_best_positions[i] - individual) +
                        self.social_coeff * r2 * (self.global_best_position - individual)
                    )
                    population[i] += velocities[i]
                    population[i] = np.clip(population[i], lb, ub)

                    score = func(population[i])
                    evaluations += 1

                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = population[i]

                        if score < self.global_best_score:
                            self.global_best_score = score
                            self.global_best_position = population[i]

                # DE-like crossover for non-elites
                for i in range(len(population)):
                    if i < elite_count:
                        continue

                    indices = list(range(len(population)))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    trial_vector = np.copy(population[i])

                    j_rand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < self.CR or j == j_rand:
                            trial_vector[j] = population[a][j] + self.F * (population[b][j] - population[c][j])
                            trial_vector[j] = np.clip(trial_vector[j], lb[j], ub[j])

                    trial_score = func(trial_vector)
                    evaluations += 1

                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector

                        if trial_score < self.global_best_score:
                            self.global_best_score = trial_score
                            self.global_best_position = trial_vector

            # Adjust CR and F based on global progress
            progress = evaluations / self.budget
            self.CR = 0.9 - 0.5 * progress
            self.F = 0.8 + 0.2 * progress

        return self.global_best_position