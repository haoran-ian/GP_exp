import numpy as np

class AdaptiveMultiSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_count = 3
        self.population_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.CR = 0.9
        self.F = 0.8
        self.elite_fraction = 0.1
        self.population = [None] * self.swarm_count
        self.velocities = [None] * self.swarm_count
        self.personal_best_positions = [None] * self.swarm_count
        self.personal_best_scores = [None] * self.swarm_count
        self.global_best_positions = [None] * self.swarm_count
        self.global_best_scores = [np.inf] * self.swarm_count

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        evaluations = 0

        for s in range(self.swarm_count):
            self.population[s] = np.random.uniform(lb, ub, (self.population_size, self.dim))
            self.velocities[s] = np.random.uniform(-1, 1, (self.population_size, self.dim))
            self.personal_best_positions[s] = np.copy(self.population[s])
            self.personal_best_scores[s] = np.array([func(ind) for ind in self.population[s]])
            best_idx = np.argmin(self.personal_best_scores[s])
            self.global_best_positions[s] = self.population[s][best_idx]
            self.global_best_scores[s] = self.personal_best_scores[s][best_idx]
            evaluations += self.population_size

        while evaluations < self.budget:
            for s in range(self.swarm_count):
                elite_count = max(1, int(self.elite_fraction * self.population_size))
                sorted_indices = np.argsort(self.personal_best_scores[s])
                elites = self.population[s][sorted_indices[:elite_count]]

                # Dynamic adaptation of inertia weight
                dynamic_inertia = self.inertia_weight * (1 - evaluations/self.budget)

                # PSO update
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(2)
                    self.velocities[s][i] = (
                        dynamic_inertia * self.velocities[s][i] +
                        self.cognitive_coeff * r1 * (self.personal_best_positions[s][i] - self.population[s][i]) +
                        self.social_coeff * r2 * (self.global_best_positions[s] - self.population[s][i])
                    )
                    self.population[s][i] += self.velocities[s][i]
                    self.population[s][i] = np.clip(self.population[s][i], lb, ub)

                    score = func(self.population[s][i])
                    evaluations += 1

                    if score < self.personal_best_scores[s][i]:
                        self.personal_best_scores[s][i] = score
                        self.personal_best_positions[s][i] = self.population[s][i]

                        if score < self.global_best_scores[s]:
                            self.global_best_scores[s] = score
                            self.global_best_positions[s] = self.population[s][i]

                # DE mutation and crossover
                for i in range(self.population_size):
                    if i < elite_count:
                        continue

                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    trial_vector = np.copy(self.population[s][i])

                    j_rand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < self.CR or j == j_rand:
                            trial_vector[j] = self.population[s][a][j] + self.F * (self.population[s][b][j] - self.population[s][c][j])
                            trial_vector[j] = np.clip(trial_vector[j], lb[j], ub[j])

                    trial_score = func(trial_vector)
                    evaluations += 1

                    if trial_score < self.personal_best_scores[s][i]:
                        self.personal_best_scores[s][i] = trial_score
                        self.personal_best_positions[s][i] = trial_vector

                        if trial_score < self.global_best_scores[s]:
                            self.global_best_scores[s] = trial_score
                            self.global_best_positions[s] = trial_vector

            # Self-adaptive control of F and CR based on progress
            if evaluations / self.budget > 0.5:
                self.CR = max(0.7, self.CR * (1 - 0.01))
                self.F = min(0.9, self.F * (1 + 0.01))

        # Select the best global best position among all swarms
        overall_best_idx = np.argmin(self.global_best_scores)
        return self.global_best_positions[overall_best_idx]