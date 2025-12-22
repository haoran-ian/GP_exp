import numpy as np

class EnhancedHybridPSO_SA_AdaptiveV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(max(5, dim // 2), 50)
        self.w_min = 0.2
        self.w_max = 1.0
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.c1_final = 0.5
        self.c2_final = 2.5
        self.temp_initial = 15
        self.temp_final = 0.01
        self.cooling_rate = 0.88
        self.positions = None
        self.velocities = None
        self.best_individual_positions = None
        self.best_individual_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.function_evaluations = 0
        self.base_mutation_rate = 0.03

    def __call__(self, func):
        np.random.seed(42)
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        self.positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.best_individual_positions = self.positions.copy()
        self.best_individual_scores = np.array([func(position) for position in self.positions])
        self.function_evaluations += self.num_particles

        self.global_best_position = self.best_individual_positions[np.argmin(self.best_individual_scores)]
        self.global_best_score = np.min(self.best_individual_scores)

        while self.function_evaluations < self.budget:
            t = self.function_evaluations / self.budget
            c1 = self.c1_final + 0.5 * np.sin(2 * np.pi * t) * (self.c1_initial - self.c1_final)
            c2 = self.c2_final + 0.5 * np.cos(2 * np.pi * t) * (self.c2_initial - self.c2_final)
            self.w = self.w_max - ((self.w_max - self.w_min) * (1 - np.exp(-5 * t)))

            # Calculate diversity
            diversity = np.mean(np.std(self.positions, axis=0))
            self.mutation_rate = self.base_mutation_rate + (0.15 * diversity)

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (self.w * self.velocities[i] +
                                      c1 * r1 * (self.best_individual_positions[i] - self.positions[i]) +
                                      c2 * r2 * (self.global_best_position - self.positions[i]))
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)

                # Boundary reflection
                self.positions[i] = np.where(self.positions[i] < lb, lb + (lb - self.positions[i]), self.positions[i])
                self.positions[i] = np.where(self.positions[i] > ub, ub - (self.positions[i] - ub), self.positions[i])

                score = func(self.positions[i])
                self.function_evaluations += 1

                if score < self.best_individual_scores[i]:
                    self.best_individual_scores[i] = score
                    self.best_individual_positions[i] = self.positions[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

            temp = self.temp_initial * ((self.cooling_rate + np.random.rand() * 0.05) ** (t))
            if temp > self.temp_final:
                for i in range(self.num_particles):
                    candidate_position = self.positions[i] + np.random.normal(0, temp * (1 - t), self.dim)
                    candidate_position = np.clip(candidate_position, lb, ub)
                    candidate_score = func(candidate_position)
                    self.function_evaluations += 1

                    if candidate_score < self.best_individual_scores[i] or \
                       np.random.rand() < np.exp(-(candidate_score - self.best_individual_scores[i]) / temp):
                        self.positions[i] = candidate_position
                        self.best_individual_scores[i] = candidate_score
                        self.best_individual_positions[i] = candidate_position

                        if candidate_score < self.global_best_score:
                            self.global_best_score = candidate_score
                            self.global_best_position = candidate_position

            # Adaptive Mutation Strategy
            if np.random.rand() < self.mutation_rate:
                mutation_percentage = 0.05 * (1 - np.exp(-10 * t))  # Adjusted mutation percentage strategy
                mutation_vector = np.random.uniform(-mutation_percentage, mutation_percentage, self.dim)
                for i in range(self.num_particles):
                    mutated_position = self.positions[i] + mutation_vector
                    mutated_position = np.clip(mutated_position, lb, ub)
                    mutated_score = func(mutated_position)
                    self.function_evaluations += 1

                    if mutated_score < self.best_individual_scores[i]:
                        self.best_individual_scores[i] = mutated_score
                        self.best_individual_positions[i] = mutated_position

                        if mutated_score < self.global_best_score:
                            self.global_best_score = mutated_score
                            self.global_best_position = mutated_position

        return self.global_best_position, self.global_best_score